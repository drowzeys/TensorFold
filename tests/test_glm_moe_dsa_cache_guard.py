"""Full GLM-5.3's cache guard (runner._check_cache_fits) on a CPU, no weights: it counts every byte the caches take
(fused.State: latent rows, index keys, the MTP layer's) plus the indexer score scratch that grows with the context,
and when any rank is short every rank refuses before any allocates."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.glm_moe_dsa.cuda import fused, runner  # noqa: E402

GIB = 1 << 30


def _w(layers=78, indexers=21, mtp=False, dcp=1, rank=0, world=1, comm=None, small=False):
    """GLM-5.3's cache widths (latent 512 + RoPE 64, index keys 128) unless ``small``; the 3.38 bpw EXL3 pack carries an
    indexer on 21 of its 78 layers."""
    cfg = SimpleNamespace(kv_lora_rank=16 if small else 512, qk_rope_head_dim=8 if small else 64,
                          index_head_dim=4 if small else 128)
    every = max(1, layers // max(indexers, 1))
    ls = [SimpleNamespace(index=i, indexer={} if indexers and i % every == 0 and i // every < indexers else None)
          for i in range(layers)]
    return SimpleNamespace(cfg=cfg, layers=ls, mtp=object() if mtp else None, dcp=dcp, device="cpu", rank=rank,
                           world=world, comm=comm)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("dcp,slots", [(1, 1), (4, 1), (1, 3)])
def test_bytes_for_is_what_state_allocates(mtp, dcp, slots):
    w = _w(layers=7, indexers=3, mtp=mtp, dcp=dcp, small=True)
    st = fused.State(w, 1001, slots)
    assert sum(1 for L in w.layers if L.indexer is not None) == 3
    assert fused.State.bytes_for(w, 1001, slots) == st.nbytes()


@pytest.mark.parametrize("kv", ["int8", "int4"])
@pytest.mark.parametrize("mtp", [False, True])
def test_bytes_for_counts_the_quantized_cache(kv, mtp):
    """--kv-dtype int8 / int4: codes, fp16 scales and the bf16 RoPE of every latent row, as State allocates them."""
    w = _w(layers=7, indexers=3, mtp=mtp)
    w.cfg = SimpleNamespace(kv_lora_rank=64, qk_rope_head_dim=8, index_head_dim=4)
    w.kv_dtype = kv
    assert fused.State.bytes_for(w, 1001) == fused.State(w, 1001).nbytes()


def test_int4_at_262k_on_glm53():
    """78 x 416 + 21 x 256 B a token a rank: 9.234 GiB at 262,144 tokens, DCP 1."""
    w = _w()
    w.kv_dtype = "int4"
    assert fused.State.bytes_for(w, 262144 - 1) == (78 * 416 + 21 * 256) * 262144
    assert round(fused.State.bytes_for(w, 262144 - 1) / GIB, 3) == 9.234


def _scores(capacity, vrows=8, prompt_rows=2048, overlap=True):
    cols = max(fused.bucket(capacity, 2048) or 0, 1)
    out = [(vrows, cols), (prompt_rows, capacity)]
    return out + ([(prompt_rows - prompt_rows // 2, capacity)] if overlap else [])


def test_per_token_bytes_on_glm53():
    """97,344 B a token a rank at DCP 1 on the 3.38 bpw pack: latent 89,856 + index keys 5,376 + score scratch ~2,112."""
    w = _w()
    cap = 262145
    state = fused.State.bytes_for(w, cap)
    assert state == (78 * 576 + 21 * 128) * 2 * (cap + 1)
    scratch = 8 * sum(fused.Buffers.score_len(w, r, c) for r, c in _scores(cap))
    assert 2000 < scratch / cap < 2200


def _free(monkeypatch, gib_by_rank):
    """torch.cuda.mem_get_info for the calling thread's rank (thread name 'rank<r>'; rank 0 otherwise)."""
    def mem_get_info():
        name = threading.current_thread().name
        r = int(name[4:]) if name.startswith("rank") else 0
        return int(gib_by_rank[r] * GIB), 128 * GIB

    monkeypatch.setattr(torch.cuda, "mem_get_info", mem_get_info)


def test_one_rank_counts_index_keys_and_scratch(monkeypatch):
    """Latent rows alone fit in what is free less the 6 GiB reserve; with the index keys and scratch they do not."""
    w, cap = _w(), 262145
    latent = 78 * 576 * 2 * (cap + 1)
    _free(monkeypatch, [latent / GIB + 6 + 0.3])
    with pytest.raises(RuntimeError, match="too little on rank 0"):
        runner.Runner._check_cache_fits(w, cap, 1, _scores(cap))


def test_one_rank_counts_the_score_scratch(monkeypatch):
    """The caches fit with half the score scratch to spare, the scratch itself does not."""
    w, cap = _w(), 262145
    scratch = 8 * sum(fused.Buffers.score_len(w, r, c) for r, c in _scores(cap))
    _free(monkeypatch, [(fused.State.bytes_for(w, cap) + scratch / 2) / GIB + 6])
    with pytest.raises(RuntimeError, match="too little on rank 0"):
        runner.Runner._check_cache_fits(w, cap, 1, _scores(cap))
    runner.Runner._check_cache_fits(w, cap, 1, [])         # the same free memory holds the caches alone


class _Comm:
    """all_gather between rank threads on the CPU."""

    def __init__(self, world):
        self.world, self.slots, self.barrier = world, [None] * world, threading.Barrier(world)

    def all_gather(self, send, recv):
        rank = int(threading.current_thread().name[4:])
        self.slots[rank] = send.clone()
        self.barrier.wait()
        n = send.numel()
        for r in range(self.world):
            recv.view(-1)[r * n:(r + 1) * n].copy_(self.slots[r].view(-1))
        self.barrier.wait()


def _four_ranks(cap):
    comm, out = _Comm(4), {}

    def run(r):
        try:
            runner.Runner._check_cache_fits(_w(rank=r, world=4, comm=comm), cap, 1, _scores(cap))
            out[r] = "ok"
        except RuntimeError as exc:
            out[r] = str(exc)

    ts = [threading.Thread(target=run, args=(r,), name=f"rank{r}") for r in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    return [out[r] for r in range(4)]


def test_262k_at_dcp1_refuses_on_every_rank(monkeypatch):
    """Device free before the caches at --context 262144, TF_GLM53_DCP=1 (four Sparks, 2026-10-05): the latent-only
    guard refused ranks 0 and 1 and let ranks 2 and 3 allocate, which swapped. Now no rank allocates."""
    _free(monkeypatch, [27.6, 27.8, 29.6, 30.1])
    got = _four_ranks(262145)
    assert all("too little on ranks 0, 1, 2" in g for g in got), got


def test_one_short_rank_stops_all_four(monkeypatch):
    _free(monkeypatch, [60.0, 60.0, 60.0, 20.0])
    got = _four_ranks(262145)
    assert all("too little on rank 3" in g for g in got), got


def test_160k_at_dcp1_still_fits(monkeypatch):
    """The served long-context setting (--context 163840, DCP 1) with the free memory those ranks had."""
    _free(monkeypatch, [27.0, 27.7, 30.5, 31.4])
    assert _four_ranks(163841) == ["ok"] * 4


def _tiny_weights(world=1):
    """A Weights-shaped object small enough for the Runner to allocate its caches and buffers on the CPU."""
    cfg = SimpleNamespace(kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8, index_head_dim=4,
                          index_topk=64, index_n_heads=2, hidden_size=32, q_lora_rank=16, intermediate_size=64,
                          moe_intermediate_size=16, n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=2)
    w = _w(layers=5, indexers=2, small=True, world=world, comm=SimpleNamespace(all_reduce=None, all_to_all=None))
    w.cfg = cfg
    w.layers[0].kv_a = SimpleNamespace(n=32)
    w.heads, w.tap_slot, w.expert_shape = 2, {1: 0}, None
    w.lm_head = torch.zeros((16, cfg.hidden_size), dtype=torch.bfloat16)
    return w


@pytest.mark.parametrize("k,world", [(0, 1), (2, 1), (0, 4)])
def test_guard_counts_the_runners_own_scratch(monkeypatch, k, world):
    """Whatever buffers the Runner builds (verify, MTP, prompt chunk and its second half), the guard counted every
    score scratch among them, and the caches."""
    seen = {}

    def check(w, capacity, slots, scores=()):
        seen["need"] = fused.State.bytes_for(w, capacity, slots) + 8 * sum(fused.Buffers.score_len(w, r, c)
                                                                             for r, c in scores)

    monkeypatch.setattr(runner.Runner, "_check_cache_fits", staticmethod(check))
    monkeypatch.setattr(runner, "PROMPT_ROWS", 256)
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **kw: None)      # the overlap's side stream (no CUDA here)
    w = _tiny_weights(world)
    rn = runner.Runner(w, 1000, k, graphs=False)
    bufs = [b for b in (rn.vb, rn.mb, rn.pb, getattr(rn, "pb1", None)) if b is not None]
    assert len(bufs) == 2 + (k > 0) + (world > 1)
    assert seen["need"] == rn.st.nbytes() + sum(b.sc.numel() * b.sc.element_size() for b in bufs)
