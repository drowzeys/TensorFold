"""Full GLM-5.3's concurrent streams (docs/design/glm-moe-dsa-concurrency.md): up to N requests decoded together, each
with MTP drafts, DFlash2 drafts or both (auto), every stream's reply token-identical to the same request alone.

Each stream owns a slot of the Runner's caches (``fused.State(slots=N)``: N streams' rows back to back). A decode round
packs every live stream's [pending token, drafts] into one verify window - up to 4 MTP streams x 3 rows, or 4 DFlash2
streams x 8 rows = 32 rows. Every window of these buffers is a decode window whatever its width
(``fused.Buffers(decode=True)``: the RoCE one-shot or all-gather + rank-order reductions, the decode attention tiling
and head buffers of a lone request's windows), and the fused kernels read each row's position and cache base from
int32 device tables (``fused.Rows``) copied in before the call or graph replay, so a row computes exactly what it
computes alone. The MTP chain runs batched: the first step takes every MTP-keeping stream's backlog rows (target
hiddens of the rows its last round kept: up to 8 for an auto stream after a DFlash2 round), steps 2..k one row a
drafting stream. DFlash2 runs batched too (``dflash.MultiDrafter``: one block pass for every DFlash2 stream, a ring a
stream, one context update for every stream's kept taps). Prompts fill one chunk per round between decode rounds, with
the chunking a lone request uses (prompt chunks are not row-invariant, so the chunk boundaries must match) and the
one-stream kernels on the stream's slot (``State.view``); a DFlash2 stream's chunks also feed its drafter context.

Modes per request: serial (no drafts), MTP (``k`` drafts a round), DFlash2 (up to the block - 1 drafts, cut by the
chain's confidence; DFLASH_CFG / TF_GLM53_DFLASH_DEPTH / _CONFIDENCE as the one-stream path reads them), auto (each
round MTP or DFlash2 per stream, whichever has been emitting more tokens a second; both stay current).

Fills between layers: while other streams decode, a chunk goes TF_GLM53_FILL_LAYERS layers a step (default 8; 0: whole chunks) with a
decode round between steps, so a long fill stalls the others for a few layers' time, not a chunk's: the same rows
through the same layers and kernels (sequence-parallel, ring-overlapped or plain, as alone), only paused between
layers (the chunk's rows wait in the Runner's prompt buffers, which decode rounds never touch). A prompt filling while
no stream decodes takes whole chunks, as alone. A DFlash2 stream's chunk feeds its drafter once the chunk is complete.

Graphs: keyed by the window's shape (rows, streams, MTP step, key bucket, pick), captured on first use; the position
and base tables are static buffers, so one graph serves every position mix of that shape.

Draft cut (TF_GLM53_DRAFT_CUT, from 4 streams): the MTP chain runs at full depth, then every rank computes each
stream's chain probability from the gathered pick logits and log-sum-exps and drops the drafts past the cut from the
verify window before it runs (the window closes up; a dropped draft is a rejected one to the MTP cache).

TP: rank 0 decides (admission, fills, rounds, arms) and samples; ranks 1..3 follow its messages - ADMIT (+ prompt),
FILL (chunk, layer range; + the first token), ROUND (+ each stream's kept tokens), DONE - one fixed-size all-gather
each, and run the same GPU work. DFlash2 drafts are computed on every rank from identical gathered candidates (as the one-stream path does),
so the window layout agrees without another message. Follower ranks never sample the target, so their host work is
small and they cannot disagree with rank 0.

Credits: the follower protocol and round structure follow TensorFold's families/qwen3_5/cuda/multi.py; per-row
positions, per-stream cache bases and the multi-stream DFlash2 drafter follow MiaAI-Lab's GLM-5.3-Flash multi-stream
patches (Apache-2.0: 0027 glm-multi-dflash2, 0029 glm-multi-dsa, 0030 glm-multi-stream-engine, 0035 glm-multi-rounds,
0049 glm-multi-prefill).
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
from typing import Callable

import torch

from tensorfold.cuda.streams import Stream

from . import fused

ADMIT, ROUND, DONE, FILL = 1, 2, 3, 4      # rank 0's messages (an empty message: stop following)
MSG = 128                                  # ints a one-shot message carries (longer ones take a second all-gather)
SERIAL, MTP, DFLASH, AUTO = 0, 1, 2, 3     # a stream's mode
NONE, ARM_M, ARM_F = 0, 1, 2               # a stream's drafts this round
ITEM = 9                                   # ints a round's plan carries a stream
PROBE = int(os.environ.get("TF_GLM53_AUTO_PROBE", "16"))

# Rank checks (MiaAI-Lab GLM-5.3-Flash 0065-glm-rank-checks, Apache-2.0): every message rank 0 shares carries a
# sequence number and a checksum, so a follower that falls out of step stops with a named error instead of applying
# another round's ints; TF_GLM_MULTI_WATCHDOG_S > 0: a round (rank 0) or the handling of one message (ranks 1..3)
# that takes longer dumps every thread's stack to stderr and exits the process (code 1), so a supervisor sees a stalled
# rank instead of four ranks waiting on each other's collectives forever. 0 (default): off. Set it above the longest
# prompt chunk (a whole 8,192-row chunk alone takes ~8 s) - the single-stream path re-arms it a round, so there it
# must also exceed the longest prompt's prefill.
SEAL_MOD = 2_147_483_647                   # a message's checksum, modulo this prime
WATCHDOG_S = float(os.environ.get("TF_GLM_MULTI_WATCHDOG_S", "0") or 0)


def digest(values: list[int], seq: int) -> int:
    """A message's checksum: its ints weighted by position, plus its sequence number, modulo SEAL_MOD."""
    import numpy as np

    a = np.asarray(values, dtype=np.int64) % SEAL_MOD
    w = np.arange(1, len(a) + 1, dtype=np.int64) % SEAL_MOD
    return int((int((a * w % SEAL_MOD).sum()) + seq * 1_000_003) % SEAL_MOD)


def watch(seconds: float = WATCHDOG_S) -> None:
    """Arm (or re-arm) the stall watchdog: stacks to stderr, then exit, after ``seconds`` (<= 0: nothing)."""
    if seconds > 0:
        import faulthandler
        import sys

        try:                                              # the process's stderr (a test may swap sys.stderr)
            faulthandler.dump_traceback_later(seconds, repeat=False, file=sys.__stderr__, exit=True)
        except (ValueError, OSError, AttributeError):     # no file descriptor to write to: no watchdog
            pass


def unwatch(seconds: float = WATCHDOG_S) -> None:
    if seconds > 0:
        import faulthandler

        faulthandler.cancel_dump_traceback_later()


def _f64_ints(x: float) -> list[int]:
    bits = int.from_bytes(struct.pack("<d", float(x)), "little")
    return [bits & 0x7FFFFFFF, (bits >> 31) & 0x7FFFFFFF, bits >> 62]


def _ints_f64(a: int, b: int, c: int) -> float:
    return struct.unpack("<d", (a | (b << 31) | (c << 62)).to_bytes(8, "little"))[0]


def _pack_sampling(s) -> list[int]:
    if s is None:
        return [0] * 13
    seed = int(s.seed) & 0xFFFFFFFFFFFFFFFF
    return [seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62, *_f64_ints(s.temperature), int(s.top_k),
            *_f64_ints(s.top_p), *_f64_ints(s.min_p)]


def _unpack_sampling(v: list[int]):
    from tensorfold.engine.exact_sampling import Sampling

    t = _ints_f64(*v[3:6])
    if t <= 0:
        return None
    return Sampling((v[2] << 62) | (v[1] << 31) | v[0], t, v[6], _ints_f64(*v[7:10]), _ints_f64(*v[10:13]))
FILL_LAYERS = int(os.environ.get("TF_GLM53_FILL_LAYERS", "8"))   # layers a fill step while others decode (0: a chunk)
# chunks under this many rows fill whole even while others decode: a short prompt in steps waited ~10 decode rounds
# and queued fills stacked (4 short requests together: 4-5 s TTFT vs ~1 s whole); long chunks keep the steps
FILL_MIN_ROWS = int(os.environ.get("TF_GLM53_FILL_MIN_ROWS", "2048"))
# Batched short fills (after bertholomus' TensorFold 14b43b0's quick fills): after a round's fill step, queued
# prompts whose rest fits in TF_GLM53_QUICK_ROWS rows (default 1024, together) fill to their first token before the
# decode round instead of one prompt a round, so a burst of short requests starts decoding together (4 chats at once
# waited up to 4 rounds for their first tokens). Each prompt takes the chunks and kernels it takes alone: the same
# bits. TF_GLM53_BATCH_FILL=0: one fill step a round, as before.
BATCH_FILL = os.environ.get("TF_GLM53_BATCH_FILL", "1") != "0"
QUICK_ROWS = int(os.environ.get("TF_GLM53_QUICK_ROWS", "1024"))
# Draft cut (after bertholomus' TensorFold 145ee42): with TF_GLM53_DRAFT_CUT_STREAMS (default 4, at least 2) or more
# streams in a round, an MTP stream's drafts stop at the first one whose chain probability (the MTP head's softmax
# probabilities of its drafts so far, multiplied) falls below TF_GLM53_DRAFT_CUT (default 0.6; 0: off), and a DFlash2
# stream's chain confidence floor rises to TF_GLM53_DRAFT_CUT_DFLASH (default: the same). A concurrent round's draft
# row costs mostly the routed experts' weight reads it adds, and what a row must earn grows with the streams (bert's
# TP4 4 chat streams: 16.0 -> 12.1 rows a round, 66.3 -> 71.0 tok/s; at 2 streams the cut cost, so it waits for 4).
# Drafts only propose: replies are the same. Every rank decides from the same gathered numbers.
DRAFT_CUT = float(os.environ.get("TF_GLM53_DRAFT_CUT", "0.6") or 0)
DRAFT_CUT_DFLASH = float(os.environ.get("TF_GLM53_DRAFT_CUT_DFLASH", "") or DRAFT_CUT)
CUT_STREAMS = max(2, int(os.environ.get("TF_GLM53_DRAFT_CUT_STREAMS", "4") or 4))
# Precapture (after bertholomus' TensorFold 4016d27: a burst of 4 streams after a start paid 36 captures in its first
# rounds): prewarm also captures the round shapes it used to leave to first use - verify windows of every width with
# both picks (the draft cut makes any width), auto streams' MTP first steps and backlog writes - from the shortest
# contexts' key buckets up, at most TF_GLM53_PRECAPTURE_MAX (default 128) more graphs and TF_GLM53_PRECAPTURE_GB
# (default 2) of device memory a rank, and never within 1 GiB of the cache reserve (TF_GLM53_CACHE_RESERVE_GB) on any
# rank. 0: as before. The same graphs replay the same kernels: replies are the same.
PRECAPTURE = os.environ.get("TF_GLM53_PRECAPTURE", "1") != "0"
PRECAPTURE_MAX = int(os.environ.get("TF_GLM53_PRECAPTURE_MAX", "128"))
PRECAPTURE_GB = float(os.environ.get("TF_GLM53_PRECAPTURE_GB", "2"))


class GlmMultiDecoder:
    """The Scheduler's decoder (live/admit/round/finish/drop) on rank 0; ``follow`` on ranks 1..3."""

    def __init__(self, runner, *, rank: int, world: int, comm, limit: int, eos: tuple[int, ...],
                 sample: Callable | None = None) -> None:
        """``sample(logits [1, V], position, sampling) -> token`` (rank 0; sampling None: greedy). A DFlash2 drafter
        on the runner (``runner.drafter``) enables DFlash2 and auto requests."""
        w = runner.w
        self.runner, self.w, self.st = runner, w, runner.st
        self.N, self.k = runner.st.slots, runner.k
        self.rank, self.world, self.comm = rank, world, comm
        self.limit, self.eos, self.sample = limit, tuple(eos), sample
        self.local = runner.st.local
        N, k = self.N, self.k
        self.dr = None
        self.frows = 0                             # rows a DFlash2 stream's window may take (pending + drafts)
        if runner.drafter is not None:
            from .dflash import MultiDrafter

            self.frows = min(runner.drafter.block, fused.DECODE_ROWS // N)
            if self.frows >= 2:
                self.dr = MultiDrafter(runner.drafter, N)
            else:
                self.frows = 0
        rows = N * max(k + 1, self.frows)
        if rows > fused.DECODE_ROWS:
            raise ValueError(f"{N} streams x {max(k + 1, self.frows)} rows = {rows} > {fused.DECODE_ROWS}: a "
                             "window that wide is not a decode window (lower --parallel or --mtp-drafts)")
        self.BL = max(k + 1, self.frows if k else 0)     # MTP backlog rows a stream (auto: a DFlash2 round's kept rows)
        self.MR = N * self.BL                            # widest first MTP step
        dev = w.device
        self.rows = rows
        self.vb = fused.Buffers(w, rows, runner.cols, decode=True)
        self.mb = fused.Buffers(w, max(self.MR, rows), runner.cols, decode=True) if k else None
        D = w.cfg.hidden_size
        self.bh = torch.zeros((max(self.MR, 1), D), dtype=torch.bfloat16, device=dev)   # MTP backlog hiddens
        self.bt = torch.zeros((max(self.MR, 1),), dtype=torch.long, device=dev)         # ... and tokens
        # the round's tables: target (pos, base, ids) then per MTP step j (pos, base, last rows, verify slots), then
        # the backlog rows step 1 gathers
        R, M = rows, max(self.MR, 1)
        o = {"vpos": 0, "vbase": R, "vids": 2 * R}
        at = 3 * R
        for j in range(1, k + 1):
            o[f"mpos{j}"], o[f"mbase{j}"], o[f"last{j}"], o[f"vdst{j}"] = at, at + M, at + 2 * M, at + 2 * M + N
            at += 2 * M + 2 * N
        o["src"] = at
        at += M
        self.off, self.tab_n = o, at
        self.t64 = torch.zeros((at,), dtype=torch.long, device=dev)
        self.t32 = torch.zeros((at,), dtype=torch.int32, device=dev)
        t32 = lambda name, n: self.t32[o[name]:o[name] + n]          # noqa: E731
        self.vrows = fused.Rows(self.st, t32("vpos", R), t32("vbase", R), None, None)
        self.mrows = {j: fused.Rows(self.st, None, None, t32(f"mpos{j}", M), t32(f"mbase{j}", M))
                      for j in range(1, k + 1)}
        # the draft cut's numbers, every MTP step's (captured with it): [0] the picked draft's logit (the gathered
        # maximum, alike on every rank), [1] this rank's log-sum-exp over its vocabulary share
        self.cut_on = DRAFT_CUT > 0 and k > 0
        self.cs = torch.zeros((2, max(k, 1), N), dtype=torch.float32, device=dev)
        self.cs_all = torch.zeros((world * max(k, 1) * N,), dtype=torch.float32, device=dev)
        self.cuts = 0                             # drafts the cut dropped (MTP)
        self._agree([int(self.cut_on)], "TF_GLM53_DRAFT_CUT (on/off)")     # the step graphs differ by it
        self.views = [self.st.view(s) for s in range(N)]
        self.samp: list = [None] * N               # each slot's sampling (every rank: DFlash2 chains use the seed)
        self.free = list(range(N))
        self.streams: dict[int, Stream] = {}      # decoding
        self.filling: list[Stream] = []           # admitted, prompt chunks left (oldest first)
        self.next_id = 0
        self.broken: Exception | None = None
        self.rounds = 0
        self.seq = 0                              # messages shared so far (every rank counts the same)
        self.iteration_since: float | None = None  # rank 0: when the running round began (/health iteration_s)
        self.widest = 0                           # the widest verify window so far (rows)
        self._cfg, self._cfg_t = (7, 0.4), -1.0
        if self.dr is not None and runner.G.enabled:
            self.dr.capture()                     # every rank together (its passes gather over the ranks)
        self.fill_layers = FILL_LAYERS
        self.mid_rounds = 0                       # decode rounds run while a prompt chunk was paused between layers

    # -------------------------------------------------------------------------------------------- messages ---
    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank: one all-gather of MSG ints (a longer list: one more for the rest), headed
        by its length, sequence number and checksum (``digest``); a follower whose count or checksum disagrees stops
        with a named error (the ranks are out of step)."""
        dev = self.w.device
        H = 3                                             # length, sequence number, checksum
        seq = self.seq
        self.seq = (self.seq + 1) % SEAL_MOD
        buf = torch.zeros((MSG,), dtype=torch.long, device=dev)
        if self.rank == 0:
            n = len(values)
            head = [n, seq, digest(list(values), seq)] + list(values[:MSG - H])
            buf[:len(head)] = torch.tensor(head, dtype=torch.long)
        if self.world == 1:
            got = buf
        else:
            got = torch.empty((self.world * MSG,), dtype=torch.long, device=dev)
            self.comm.all_gather(buf, got)
        first = got[:MSG].tolist()
        n, got_seq, check = (int(v) for v in first[:H])
        if self.rank != 0 and got_seq != seq:
            raise RuntimeError(f"rank {self.rank}: expected rank 0's message {seq}, received number {got_seq} "
                               f"({n} ints): the ranks are out of step; restart all four")
        out = first[H:H + min(n, MSG - H)]
        rest = n - len(out)
        if rest > 0:
            tail = (torch.tensor(values[MSG - H:], dtype=torch.long, device=dev) if self.rank == 0
                    else torch.zeros((rest,), dtype=torch.long, device=dev))
            if self.world > 1:
                allt = torch.empty((self.world * rest,), dtype=torch.long, device=dev)
                self.comm.all_gather(tail, allt)
                tail = allt[:rest]
            out += tail.tolist()
        if self.rank != 0 and check != digest(out, seq):
            raise RuntimeError(f"rank {self.rank}: rank 0's message {seq} ({n} ints) fails its checksum: the ranks "
                               "are out of step; restart all four")
        return out

    def _agree(self, values: list[int], what: str) -> None:
        """Every rank (startup): the same settings everywhere, or a named error (a setting that changes what a rank
        captures or gathers would put the ranks' collectives out of step)."""
        if self.world == 1:
            return
        dev = self.w.device
        mine = torch.tensor(values, dtype=torch.long, device=dev)
        got = torch.empty((self.world * len(values),), dtype=torch.long, device=dev)
        self.comm.all_gather(mine, got)
        rows = got.view(self.world, -1).tolist()
        if any(r != rows[0] for r in rows):
            raise RuntimeError(f"the ranks were started with different {what}: {rows}; give every rank the same "
                               "settings")

    def _send(self, values: list[int]) -> None:
        if self.world > 1 and self.broken is None:
            self._share(values)

    def _recv(self) -> list[int]:
        return self._share(None)

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the ranks are out of step after an error; restart all four") from self.broken

    # --------------------------------------------------------------------------------------------- streams ---
    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def nbytes_per_stream(self) -> int:
        return self.st.nbytes() // self.N + (self.dr.nbytes() // self.N if self.dr is not None else 0)

    def _mode(self, draft) -> int:
        """A request's ``draft``: False (serial), True or an MTP mode (MTP drafts), "dflash" or "auto"."""
        if not draft:
            return SERIAL
        if draft in ("dflash", "auto"):
            if self.dr is None:
                raise ValueError(f"mtp mode {draft!r}: no DFlash2 drafter loaded (TF_GLM53_DFLASH)")
            return AUTO if draft == "auto" and self.k else DFLASH
        return MTP if self.k else SERIAL

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot; rounds fill its prompt a chunk at a time."""
        self._check()
        if len(s.prompt) >= self.limit:
            raise ValueError(f"prompt of {len(s.prompt)} tokens: this engine serves contexts up to {self.limit}")
        if not self.free:
            raise RuntimeError("no free stream slot (the scheduler admits at most --parallel streams)")
        mode = self._mode(s.draft)
        s.count = max(1, min(int(s.count), self.limit - len(s.prompt)))
        s.sid = self.next_id
        self.next_id += 1
        temp = getattr(s.sampling, "temperature", 0.0) if s.sampling is not None else 0.0
        if temp <= 0:
            s.sampling = None
        slot = self.free.pop(0)
        self._send([ADMIT, s.sid, slot, mode, int(s.sampling is not None), *_pack_sampling(s.sampling)])
        self._send(list(s.prompt))
        self._queue(s, slot, mode)

    def _queue(self, s: Stream, slot: int, mode: int) -> None:
        s.slot, s.mode = slot, mode
        s.chunks = self.runner.chunks(len(s.prompt))
        s.ci = 0
        s.li = 0                                  # the current chunk's next layer
        s.toks = torch.tensor(s.prompt, dtype=torch.long, device=self.w.device)
        s.ema, s.since = {"m": None, "f": None}, {"m": 0, "f": 0}
        self.samp[slot] = s.sampling
        if self.dr is not None:
            self.dr.reset(slot)
        self.filling.append(s)

    def _keeps_mtp(self, mode: int) -> bool:
        return mode in (MTP, AUTO) and self.k > 0

    def _taps(self, mode: int) -> bool:
        return mode in (DFLASH, AUTO) and self.dr is not None

    # ------------------------------------------------------------------------------------------------ fill ---
    def _fill(self, alone: bool) -> list[Stream]:
        """The oldest queued prompt's next step: the rest of its current chunk (the chunking it has alone) when no
        stream decodes (``alone``), else the chunk's next fill_layers layers; at the prompt's end, its first token."""
        s = self.filling[0]
        a, e = s.chunks[s.ci]
        n, G = len(self.w.layers), self.fill_layers
        hi = n if alone or G <= 0 or e - a < FILL_MIN_ROWS else min(n, s.li + G)
        self._send([FILL, s.sid, a, e, s.li, hi])
        t0 = time.perf_counter()
        try:
            first = self._chunk(s, a, e, s.li, hi)
            if first is not None:
                tok = int(self.sample(first, len(s.prompt), s.sampling))
                self._send([tok])
                self._start(s, tok)
        except Exception as exc:
            if self.world > 1:
                self.broken = exc
            raise
        finally:
            s.prefill_s += time.perf_counter() - t0
        if first is None:
            return []
        s.take([tok], self._ends(s))
        return [s] if s.done else []

    def _quick_fills(self) -> list[Stream]:
        """BATCH_FILL: after the round's fill step, the queued prompts whose rest fits in QUICK_ROWS rows (together)
        fill to their first token before the decode round, oldest first, each in the chunks and kernels it takes alone
        (one ``_fill`` a chunk: no other prompt's step comes between a prompt's chunks, as before, so the Runner's
        prompt buffers and MTP carry stay that prompt's)."""
        done, used = [], 0
        while self.filling:
            s = self.filling[0]
            a = s.chunks[s.ci][0]
            rest = len(s.prompt) - a
            if used + rest > QUICK_ROWS:
                break
            used += rest
            while self.filling and self.filling[0] is s:
                done += self._fill(not any(not x.done for x in self.streams.values()))
        return done

    def _chunk(self, s: Stream, a: int, e: int, lo: int, hi: int):
        """Layers [lo, hi) of chunk a..e (all of them: the chunk in one go, as alone)."""
        n = len(self.w.layers)
        if lo != s.li:
            raise RuntimeError(f"fill step at layer {lo}, the chunk is at layer {s.li}")
        layers = None if (lo, hi) == (0, n) else (lo, hi)
        out = self.runner.prefill_chunk(self.views[s.slot], s.toks, a, e, len(s.prompt), layers=layers)
        if hi < n:
            s.li = hi
            return None
        s.li = 0
        s.ci += 1
        if self._taps(s.mode):                    # the drafter's context: this chunk's committed taps
            self.dr.commit([(s.slot, self.runner.pb.taps[:e - a])])
        return out

    def _start(self, s: Stream, tok: int) -> None:
        """Prompt done: the pending token at P = len(prompt); the MTP backlog (carry = hidden P - 1, token)."""
        L0 = len(s.prompt)
        s.P, s.tok, s.m = L0, tok, 1
        if self.k:
            i = s.slot * self.BL
            self.bh[i].copy_(self.runner.carry)
            self.bt[i:i + 1].fill_(tok)
        s.toks = None
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s

    # ----------------------------------------------------------------------------------------------- rounds ---
    def _dflash_cfg(self) -> tuple[int, float]:
        """Rank 0: (depth, confidence) as the one-stream path reads them (env, then the DFLASH_CFG file; re-read at
        most once a second), the depth capped by the rows a stream's window may take."""
        now = time.perf_counter()
        if now - self._cfg_t >= 1.0:
            self._cfg_t = now
            from .runner import PROFILE_FLAG

            cfg = {"depth": int(os.environ.get("TF_GLM53_DFLASH_DEPTH", "7")),
                   "confidence": float(os.environ.get("TF_GLM53_DFLASH_CONFIDENCE", "0.3"))}
            try:
                cfg.update(json.loads(open(os.path.dirname(PROFILE_FLAG) + "/DFLASH_CFG").read()))
            except (OSError, ValueError):
                pass
            self._cfg = (int(cfg["depth"]), float(cfg["confidence"]))
        depth, conf = self._cfg
        return min(self.runner.drafter.block - 1, self.frows - 1, depth), conf

    def _arm(self, s: Stream) -> int:
        """Rank 0: this round's drafts for a decoding stream (auto: the arm emitting more tokens a second, with a
        probe of the other every PROBE rounds - the one-stream auto rule, per stream)."""
        if s.mode == MTP:
            return ARM_M
        if s.mode == DFLASH:
            return ARM_F
        if s.mode != AUTO:
            return NONE
        ema, since = s.ema, s.since
        if ema["m"] is None or ema["f"] is None:
            arm = "m" if ema["m"] is None else "f"
        else:
            arm = "m" if ema["m"] >= ema["f"] else "f"
            other = "f" if arm == "m" else "m"
            if since[other] >= PROBE:
                arm = other
        since[arm] = 0
        since["f" if arm == "m" else "m"] += 1
        s.last_arm = arm
        return ARM_M if arm == "m" else ARM_F

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """A prompt chunk for the oldest queued prompt, then one decode round over the decoding streams; returns the
        streams that finished."""
        self._check()
        self.iteration_since = time.monotonic()
        watch()                                          # TF_GLM_MULTI_WATCHDOG_S: a stalled round exits
        try:
            return self._round()
        finally:
            unwatch()
            self.iteration_since = None

    def _round(self) -> list[Stream]:
        decoding = any(not s.done for s in self.streams.values())
        done = self._fill(not decoding) if self.filling else []
        if BATCH_FILL:
            done += self._quick_fills()
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        if self.filling and self.filling[0].li:
            self.mid_rounds += 1
        t0 = time.perf_counter()
        depth, conf = self._dflash_cfg() if self.dr is not None else (0, 0.0)
        cut = 0.0
        if len(live) >= CUT_STREAMS:                     # the draft cut (rank 0's numbers in the message)
            cut = DRAFT_CUT if self.cut_on else 0.0
            if self.dr is not None and DRAFT_CUT_DFLASH > 0:
                conf = max(conf, DRAFT_CUT_DFLASH)
        cap = self.runner.capacity
        plan = []
        for s in live:
            arm = self._arm(s)
            room = (self.k if arm == ARM_M else
                    max(0, min(depth, s.count - len(s.out) - 1, cap - s.P - 1)) if arm == ARM_F else 0)
            plan.append((s.sid, s.slot, s.P, s.tok, s.m, s.mode, arm, room, int(s.sampling is not None)))
        self._send([ROUND, len(plan), *_f64_ints(conf), *_f64_ints(cut), *[x for item in plan for x in item]])
        try:
            offs, R, ds = self._gpu(plan, conf, cut)
            results = self._picks(plan, live, offs, ds)
            self._send([x for n, emit in results for x in (n, len(emit), *emit)])
            self._commit(plan, offs, results)
        except Exception as exc:
            if self.world > 1:
                self.broken = exc
            raise
        self.rounds += 1
        dt = time.perf_counter() - t0
        for s, item, d, (n, emit) in zip(live, plan, ds, results):
            s.counted(1 + d)
            new = []
            for t in emit:                                   # a lone request's room: its count, its end tokens
                if len(s.out) + len(new) >= s.count:
                    break
                new.append(t)
                if t in self._ends(s):
                    break
            s.P, s.tok, s.m = s.P + n + 1, emit[-1], n + 1
            if s.mode == AUTO:
                arm = s.last_arm
                rate = (n + 1) / max(dt, 1e-6)
                s.ema[arm] = rate if s.ema[arm] is None else 0.8 * s.ema[arm] + 0.2 * rate
                s.arms = getattr(s, "arms", "") + arm
            s.take(new, self._ends(s))
        return done + [s for s in live if s.done]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    def _gpu(self, plan, conf: float, cut: float = 0.0) -> tuple[list[int], int, list[int]]:
        """Every rank: the DFlash2 drafts, the round's tables, the batched MTP chain, the verify window. Returns row
        offsets, R and each stream's drafted rows. ``cut`` > 0: the MTP drafts past the cut leave the verify window
        (``_cut``) before it runs."""
        w, rn, k, o = self.w, self.runner, self.k, self.off
        vb = self.vb
        drafts: dict[int, list[int]] = {}
        reqs = [(i, (slot, tok, room, self.samp[slot], conf))
                for i, (sid, slot, P, tok, m, mode, arm, room, _) in enumerate(plan) if arm == ARM_F and room]
        if reqs:
            got = self.dr.propose([r for _, r in reqs])
            drafts = {i: d for (i, _), d in zip(reqs, got)}
        tab = [0] * self.tab_n
        offs, ds, r = [], [], 0
        for i, (sid, slot, P, tok, m, mode, arm, room, _) in enumerate(plan):
            d = room if arm == ARM_M else len(drafts.get(i, ()))
            offs.append(r)
            ds.append(d)
            for j in range(1 + d):
                tab[o["vpos"] + r + j] = P + j
                tab[o["vbase"] + r + j] = slot * self.local
            tab[o["vids"] + r] = tok
            for j, t in enumerate(drafts.get(i, ())):
                tab[o["vids"] + r + 1 + j] = t
            r += 1 + d
        R = r
        keep = [(q, item, off) for q, (item, off) in enumerate(zip(plan, offs)) if self._keeps_mtp(item[5])]
        S = M = 0
        Tm = None
        drafting = []                                    # plan index of each MTP-drafting stream (step row order)
        if keep:
            for q, (sid, slot, P, tok, m, mode, arm, room, _), off in keep:
                for i in range(m):
                    tab[o["mpos1"] + M] = P - m + i
                    tab[o["mbase1"] + M] = slot * self.local
                    tab[o["src"] + M] = slot * self.BL + i
                    M += 1
                if arm != ARM_M or not room:
                    continue
                drafting.append(q)
                tab[o["last1"] + S] = M - 1
                tab[o["vdst1"] + S] = off + 1
                for j in range(2, k + 1):
                    tab[o[f"mpos{j}"] + S] = P + j - 2
                    tab[o[f"mbase{j}"] + S] = slot * self.local
                    tab[o[f"last{j}"] + S] = S
                    tab[o[f"vdst{j}"] + S] = off + j
                S += 1
            Tm = rn._T(max(item[2] for _, item, _ in keep) + k)
        self.t64.copy_(torch.tensor(tab, dtype=torch.long))
        self.t32.copy_(self.t64)
        vb.ids[:R].copy_(self.t64[o["vids"]:o["vids"] + R])
        if M:
            mb = self.mb
            src = self.t64[o["src"]:o["src"] + M]
            torch.index_select(self.bh, 0, src, out=mb.hin[:M])
            torch.index_select(self.bt, 0, src, out=mb.ids[:M])
            if S:
                for j in range(1, k + 1):
                    self._mtp_step(j, M if j == 1 else S, S, Tm)
            else:                                        # auto streams on DFlash2 rounds: their backlog only
                self._mtp_write(M, Tm)
        if cut > 0 and S:
            offs, ds, R = self._cut(plan, offs, ds, drafting, cut, R)
        self.widest = max(self.widest, R)
        T = rn._T(max(P + 1 + d for (_, _, P, *_), d in zip(plan, ds)))
        pick = "full" if any(item[8] for item in plan) else "argmax"
        self._verify(R, T, pick)
        return offs, R, ds

    def _verify(self, R: int, T: int | None, pick: str) -> None:
        w, vb, rows = self.w, self.vb, self.vrows
        self.runner.G.run(("mt", R, T, pick), lambda: fused.compute(w, rows, vb, R, T, logits="all", pick=pick))

    def _mtp_step(self, j: int, n: int, S: int, Tm: int | None) -> None:
        w, rn, mb, vb, o = self.w, self.runner, self.mb, self.vb, self.off
        cn, full = rn.chain_normed, rn.draft_full
        rows = self.mrows[j]
        last = self.t64[o[f"last{j}"]:o[f"last{j}"] + S]
        vdst = self.t64[o[f"vdst{j}"]:o[f"vdst{j}"] + S]

        def fn():
            fused.mtp_compute(w, rows, mb, n, Tm, chain_normed=cn, draft_full=full, last=last)
            if self.cut_on:
                self._draft_stats(j, S, full)
            vb.ids.index_copy_(0, vdst, mb.argmax[:S])
            mb.ids[:S].copy_(mb.argmax[:S])
            torch.index_select(mb.hidden, 0, last, out=mb.hin[:S])
        rn.G.run(("mm", n, S, j, Tm, cn, full), fn)

    def _draft_stats(self, j: int, S: int, full: bool) -> None:
        """MTP step j's S picks: their logit (the maximum the head gathered from every rank) and this rank's
        log-sum-exp over its share of the head's vocabulary (still in mb.lpart), into self.cs[:, j - 1, :S]."""
        w, mb = self.w, self.mb
        from . import headq

        table = w.draft_lm_head if full or w.draft_head is None else w.draft_head    # fused.head's draft table
        V = headq.rows_of(table)
        lg = mb.lpart.view(-1)[:S * V].view(S, V)
        self.cs[1, j - 1, :S].copy_(torch.logsumexp(lg, dim=-1))
        if w.world > 1:
            g = mb.amax_all[:w.world * S * 4].view(w.world, S, 4)
            self.cs[0, j - 1, :S].copy_(g[:, :, 0].amax(dim=0))
        else:
            self.cs[0, j - 1, :S].copy_(mb.amax[:S, 0])

    def _cut(self, plan, offs, ds, drafting, cut: float, R: int) -> tuple[list[int], list[int], int]:
        """Every rank, after the MTP chain: each drafting stream keeps its drafts while their chain probability stays
        at or above ``cut``; the rest leave the verify window, which closes up (positions, bases, ids). The
        probabilities come from numbers every rank holds alike - the gathered pick logits and every rank's
        log-sum-exp, gathered here - so every rank cuts the same rows. Drafts only propose: the replies keep."""
        k, N, o, vb = self.k, self.N, self.off, self.vb
        S = len(drafting)
        if self.world > 1:
            self.comm.all_gather(self.cs[1].reshape(-1), self.cs_all)
            lse = self.cs_all.view(self.world, k, N)[:, :, :S]
        else:
            lse = self.cs[1:2, :, :S]
        flat = torch.cat([self.cs[0, :, :S].reshape(-1), lse.reshape(-1)]).tolist()
        pick, parts = flat[:k * S], flat[k * S:]
        ranks = len(parts) // (k * S)
        new = list(ds)
        for s, q in enumerate(drafting):
            chain, keep = 1.0, 0
            for j in range(min(k, ds[q])):
                ls = [parts[r * k * S + j * S + s] for r in range(ranks)]
                top = max(ls)
                total = top + math.log(sum(math.exp(x - top) for x in ls))
                chain *= math.exp(min(0.0, pick[j * S + s] - total))
                if chain < cut:
                    break
                keep = j + 1
            new[q] = keep
        if new == list(ds):
            return offs, ds, R
        self.cuts += sum(ds) - sum(new)
        src, pos, base, noffs, r = [], [], [], [], 0
        for (sid, slot, P, *_), off, d in zip(plan, offs, new):
            noffs.append(r)
            for j in range(1 + d):
                src.append(off + j)
                pos.append(P + j)
                base.append(slot * self.local)
            r += 1 + d
        t = torch.tensor(pos + base + src, dtype=torch.long).to(self.w.device)
        for name, part in (("vpos", t[:r]), ("vbase", t[r:2 * r])):
            self.t64[o[name]:o[name] + r].copy_(part)
            self.t32[o[name]:o[name] + r].copy_(part)
        vb.ids[:r].copy_(vb.ids[:R].index_select(0, t[2 * r:]))
        return noffs, new, r

    def _mtp_write(self, n: int, Tm: int | None) -> None:
        """The MTP layer's cache at backlog rows only (no drafts this round)."""
        w, rn, mb = self.w, self.runner, self.mb
        cn, rows = rn.chain_normed, self.mrows[1]
        rn.G.run(("mw", n, Tm, cn), lambda: fused.mtp_compute(w, rows, mb, n, Tm, logits="none", chain_normed=cn))

    def _picks(self, plan, live, offs, ds) -> list[tuple[int, list[int]]]:
        """Rank 0: each stream's target picks along its drafts -> (drafts kept, tokens emitted)."""
        vb = self.vb
        R = offs[-1] + 1 + ds[-1]
        both = torch.cat([vb.argmax[:R], vb.ids[:R]]).tolist()
        amax, ids = both[:R], both[R:]
        out = []
        for s, item, off, d in zip(live, plan, offs, ds):
            P, sampled = item[2], item[8]
            drafts = ids[off + 1:off + 1 + d]
            if not sampled:
                picks = amax[off:off + 1 + d]
            else:
                picks = []
                for i in range(1 + d):
                    picks.append(int(self.sample(vb.logits[off + i:off + i + 1], P + i + 1, s.sampling)))
                    if i >= d or drafts[i] != picks[-1]:
                        break
            n = 0
            while n < d and n < len(picks) - 1 and drafts[n] == picks[n]:
                n += 1
            out.append((n, picks[:n + 1]))
        return out

    def _commit(self, plan, offs, results) -> None:
        """Every rank: each MTP-keeping stream's next backlog (the kept rows' target hiddens and tokens), each
        DFlash2 stream's kept taps into its drafter context."""
        if self.dr is not None:
            items = [(item[1], self.vb.taps[off:off + n + 1])
                     for item, off, (n, _) in zip(plan, offs, results) if self._taps(item[5])]
            if items:
                self.dr.commit(items)
        if not self.k:
            return
        idx, dst, toks = [], [], []
        for (sid, slot, P, tok, m, mode, arm, room, _), off, (n, emit) in zip(plan, offs, results):
            if not self._keeps_mtp(mode):
                continue
            for i in range(n + 1):
                idx.append(off + i)
                dst.append(slot * self.BL + i)
                toks.append(emit[i])
        if not idx:
            return
        c = len(idx)
        t = torch.tensor(idx + dst + toks, dtype=torch.long).to(self.w.device)
        src = self.vb.hidden.index_select(0, t[:c])
        if self.runner.hid_normed:
            out = self.vb.normed[:c]
            from tensorfold.families.glm5_next.cuda import glue

            glue.rmsnorm(src, self.w.final_norm, self.w.cfg.rms_norm_eps, out)
        else:
            out = src
        self.bh.index_copy_(0, t[c:2 * c], out)
        self.bt.index_copy_(0, t[c:2 * c], t[2 * c:])

    @torch.no_grad()
    def prewarm(self) -> None:
        """Every rank (no messages): capture the windows of 1..N MTP-drafting streams with every backlog total
        (S..S(k+1) MTP rows), and (DFlash2 loaded) verify windows of every width up to the widest, in every key
        bucket; sampled MTP mixes capture with pick "full"; then (TF_GLM53_PRECAPTURE) the remaining round shapes within a
        memory and count budget (``_precapture``), other shapes on first use. Caches get scratch values at the
        positions used; every request writes a position before reading it."""
        self._agree([int(PRECAPTURE), PRECAPTURE_MAX, int(PRECAPTURE_GB * 1000)],
                    "TF_GLM53_PRECAPTURE / _MAX / _GB")  # the ranks' captures pair their collectives
        t0 = time.perf_counter()
        free0 = torch.cuda.mem_get_info()[0]
        rn, k = self.runner, self.k
        cap = rn.capacity - max(k, self.frows) - 2
        before = len(rn.G.graphs)
        buckets = rn.buckets()
        for i, T in enumerate(buckets):                  # P: past the previous bucket (or index_topk), so _T lands here
            P = 10 if T is None else min((buckets[i - 1] or rn.topk) + 100, cap)
            if k and rn._T(P + k + 1) == T:
                for S in range(1, self.N + 1):
                    for M in range(S, S * (k + 1) + 1):
                        ms = [M // S + (i < M % S) for i in range(S)]
                        for sampled in ((0, 1) if M == S else (0,)):
                            self._gpu([(i, i, P, 1000, ms[i], MTP, ARM_M, k, sampled) for i in range(S)], 0.0)
            if self.dr is not None and rn._T(P + self.rows) == T:
                tab = torch.zeros((self.tab_n,), dtype=torch.long)
                tab[self.off["vpos"]:self.off["vpos"] + self.rows] = torch.arange(P, P + self.rows)
                self.t64.copy_(tab)
                self.t32.copy_(self.t64)
                self.vb.ids.fill_(1000)
                for R in range(1, self.rows + 1):
                    self._verify(R, T, "argmax")
        torch.cuda.synchronize()
        free1 = torch.cuda.mem_get_info()[0]
        extra = self._precapture(buckets, cap) if PRECAPTURE and rn.G.enabled else ""
        print(f"[tensorfold] rank {self.w.rank}: {len(rn.G.graphs) - before} concurrent decode graphs captured in "
              f"{time.perf_counter() - t0:.1f}s ({(free0 - free1) / 2**30:.2f} GiB of device memory{extra})",
              flush=True)

    def _frees(self) -> list[int]:
        """Every rank's free device memory (bytes), on every rank (a collective: every rank calls it together)."""
        free = torch.cuda.mem_get_info()[0]
        if self.world == 1:
            return [free]
        dev = self.w.device
        got = torch.empty((self.world,), dtype=torch.long, device=dev)
        self.comm.all_gather(torch.tensor([free], dtype=torch.long, device=dev), got)
        return got.tolist()

    def _scratch(self, P: int, width: int) -> None:
        """Tables for scratch windows: verify row r in slot r // width at P + r % width; MTP backlog row i in slot
        i // BL at P + i % BL, gathering backlog row i; drafting stream s's last MTP row s, its draft into row s.
        Positions stay below the capacity (P <= capacity - max(k, frows) - 2)."""
        o, R, MR, BL = self.off, self.rows, self.MR, self.BL
        tab = torch.zeros((self.tab_n,), dtype=torch.long)
        r = torch.arange(R)
        tab[o["vpos"]:o["vpos"] + R] = P + r % width
        tab[o["vbase"]:o["vbase"] + R] = (r // width) * self.local
        if self.k:
            i = torch.arange(MR)
            tab[o["mpos1"]:o["mpos1"] + MR] = P + i % BL
            tab[o["mbase1"]:o["mbase1"] + MR] = (i // BL) * self.local
            tab[o["src"]:o["src"] + MR] = i
            tab[o["last1"]:o["last1"] + self.N] = torch.arange(self.N)
            tab[o["vdst1"]:o["vdst1"] + self.N] = torch.arange(self.N)
            self.mb.ids.fill_(1000)
            self.mb.hin.zero_()
        self.t64.copy_(tab)
        self.t32.copy_(self.t64)
        self.vb.ids.fill_(1000)

    def _precapture(self, buckets, cap: int) -> str:
        """PRECAPTURE (every rank, the same keys in the same order): the round shapes prewarm leaves to first use - verify
        windows of every width 1..rows with both picks (the draft cut, serial streams and sampled mixes make any width),
        and with DFlash2 and MTP both loaded (auto streams), MTP first steps of S drafting streams among more keeping
        ones and backlog-only writes - bucket by bucket from the shortest contexts, until PRECAPTURE_MAX graphs or
        PRECAPTURE_GB of device memory on any rank, or free memory on any rank near the cache reserve (GB10's unified
        memory: an overcommit swaps the node instead of failing). Shapes left over capture on first use, as before."""
        rn, k, N = self.runner, self.k, self.N
        cn, full = rn.chain_normed, rn.draft_full
        auto = self.dr is not None and k > 0
        t0 = time.perf_counter()
        start = self._frees()
        reserve = min(float(os.environ.get("TF_GLM53_CACHE_RESERVE_GB", "6")) * (1 << 30), min(start) / 2)
        budget = PRECAPTURE_GB * (1 << 30)
        made, stop, used, free = 0, "", 0, start
        for i, T in enumerate(buckets):
            P = 10 if T is None else min((buckets[i - 1] or rn.topk) + 100, cap)
            items = [(("mt", R, T, pick), lambda R=R, pick=pick, T=T: self._verify(R, T, pick))
                     for pick in ("argmax", "full") for R in range(1, self.rows + 1)]
            if auto:
                items += [(("mw", n, T, cn), lambda n=n, T=T: self._mtp_write(n, T)) for n in range(1, self.MR + 1)]
                items += [(("mm", n, S, 1, T, cn, full), lambda n=n, S=S, T=T: self._mtp_step(1, n, S, T))
                          for S in range(1, N + 1) for n in range(S, self.MR + 1)]
            items = [(key, run) for key, run in items if key not in rn.G.graphs]
            if not items:
                continue
            self._scratch(P, self.rows // N)
            for key, run in items:
                if made >= PRECAPTURE_MAX:
                    stop = f"TF_GLM53_PRECAPTURE_MAX={PRECAPTURE_MAX}"
                elif made % 8 == 0:                      # every rank at the same counts (the check is a collective)
                    free = self._frees()
                    used = max(a - b for a, b in zip(start, free))
                    if used > budget:
                        stop = f"TF_GLM53_PRECAPTURE_GB={PRECAPTURE_GB:g}"
                    elif min(free) < reserve + (1 << 30):
                        stop = f"{min(free) / 2**30:.1f} GiB free on a rank"
                if stop:
                    break
                run()
                made += 1
            if stop:
                break
        torch.cuda.synchronize()
        free = self._frees()
        used = max(a - b for a, b in zip(start, free))
        return (f"; precapture {made} more in {time.perf_counter() - t0:.1f}s, {used / 2**30:.2f} GiB on the rank that "
                f"gave most, {min(free) / 2**30:.1f} GiB free on the tightest{'; stopped at ' + stop if stop else ''}")

    # ------------------------------------------------------------------------------------------- endings ---
    def finish(self, done: list[Stream]) -> None:
        if done:
            self._send([DONE, len(done), *[s.sid for s in done]])
            for s in done:
                self._finish(s.sid)

    def _finish(self, sid: int) -> None:
        s = self.streams.pop(sid, None)
        if s is None:
            s = next((x for x in self.filling if x.sid == sid), None)
            if s is not None:
                self.filling = [x for x in self.filling if x is not s]
        if s is not None:
            s.toks = None
            self.samp[s.slot] = None
            if s.slot not in self.free:
                self.free.append(s.slot)
                self.free.sort()

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live and queued streams (the ranks may no longer agree)."""
        live = list(self.streams.values()) + self.filling
        for s in live:
            self._finish(s.sid)
        self.streams, self.filling = {}, []
        if self.world > 1 and self.broken is None:
            self.broken = RuntimeError("a round failed")
        return live

    def stop(self) -> None:
        """Rank 0: release the followers (an empty message)."""
        self._send([])

    # ------------------------------------------------------------------------------------------- followers ---
    @torch.no_grad()
    def follow(self) -> None:
        """Ranks 1..3: mirror rank 0's admissions, fills, rounds and endings until it sends an empty message."""
        while True:
            msg = self._recv()                           # idle: waits as long as rank 0 has nothing to do
            if not msg:
                return
            watch()                                      # busy: the message's work and its follow-up messages
            self._apply(msg)
            unwatch()

    def _apply(self, msg: list[int]) -> None:
        """Ranks 1..3: one message of rank 0 (and the messages that follow it), as rank 0 ran it."""
        kind = msg[0]
        if kind == ADMIT:
            sid, slot, mode, sampled = msg[1:5]
            s = Stream(self._recv(), 1, _unpack_sampling(msg[5:18]), sid=sid)
            s.sampled = bool(sampled)
            self.free = [x for x in self.free if x != slot]
            self._queue(s, slot, mode)
        elif kind == FILL:
            sid, a, e, lo, hi = msg[1:6]
            s = next(x for x in self.filling if x.sid == sid)
            if self._chunk(s, a, e, lo, hi) is not None:
                self._start(s, self._recv()[0])
        elif kind == ROUND:
            n = msg[1]
            conf, cut = _ints_f64(*msg[2:5]), _ints_f64(*msg[5:8])
            plan = [tuple(msg[8 + ITEM * i:8 + ITEM * (i + 1)]) for i in range(n)]
            offs, R, ds = self._gpu(plan, conf, cut)
            flat = self._recv()
            results, i = [], 0
            for _ in plan:
                kept, c = flat[i], flat[i + 1]
                results.append((kept, flat[i + 2:i + 2 + c]))
                i += 2 + c
            self._commit(plan, offs, results)
            self.rounds += 1
        elif kind == DONE:
            for sid in msg[2:2 + msg[1]]:
                self._finish(sid)
