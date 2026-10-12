# Several streams with the cache split over the ranks (GLM-5.3, TP4)

Decode context parallelism (DCP) is what makes a 1M-token context fit: the attention cache is sharded over the four
ranks, a quarter each. Until this branch the engine refused DCP with more than one stream, so the served shapes were
1M x 1, 64K x 2 and 32K x 4. `TF_GLM53_DCP_STREAMS=1` (every rank) lifts that: `--parallel 2 --context 500000` and
`--parallel 4 --context 250000` hold the same cache bytes as 1M x 1.

**Nothing here has run on a GPU.** The host tests cover the layout, the index math, the boot checks and the kernel
coverage; the device path needs four new kernel variants compiled into the pack and the gates below.

## Why it was refused, restriction by restriction

None of the refusals is a property of the method. The Python engine never wrote the combination, and the Zig port
mirrored it.

| Where | What it said | Why it was there | What changed |
|---|---|---|---|
| `fused.attention_part` (Python), `cuda_forward.zig` attentionPart | several streams in one window need DCP 1 | The per-row kernels (`ROWS`: a position and a cache base a row) ignore `DCP` in `_kv_write` / `_ik_write`, and `_attn_dcp` reads one window position. No kernel existed for the combination. | `*RowsDcp` launches; `_attn_dcp_rows` |
| `cuda_forward.zig` State.initSlots | slots > 1 needs DCP 1 | Follows from the above; the allocation itself never cared | removed |
| `cuda_forward.zig` select | DCP with a base table is invalid | `_index_scores` already composes `ROWS` with `DCP` in its source; no variant was compiled and the Zig side never launched it | `indexScoresRowsDcp` |
| `cuda_runner.zig` init, `cuda_multi.zig` init | BadDcp / Invalid for slots > 1 with DCP | Guards for the above | removed |
| `cuda_multi.zig` copySlotRows | copied `n` rows of a resumed state between slots | Counted rows as positions, right only at DCP 1 | copies `localRows(n)` / `localMtpRows(n)` |
| `cuda_engine.zig` checkParallel | `--parallel N` needs DCP off | Guard for the above | allowed with `TF_GLM53_DCP_STREAMS=1`, under a token cap |
| `cuda_forward.zig` attentionPart | several streams: no MTP index reuse | Reuse keeps one row's selection across an MTP chain; a concurrent chain's first step holds several rows. Unrelated to DCP. | unchanged: reuse stays off with several streams, as at DCP 1 |
| `cuda_runner.zig` / `cuda_prompt.zig` | no sequence-parallel prompt chunk under DCP, chunks of at most 1,024 rows | The sequence-parallel chunk gives each rank half the rows and their keys; under DCP every rank must see every row's query. A different algorithm, not a guard. | unchanged: prompts under DCP fill as the 1M shape fills them |

What needs no change, and why:

- **The exchanges.** Queries gathered, partials and log-sum-exps returned, candidates gathered: all are per row and
  carry no position. `_merge_lse`, `_dcp_combine`, the candidate kernels and the top-key selects are the one-stream
  launches.
- **Admission, rounds, picks, the ring of followers.** `cuda_multi` plans rounds in positions and slots. A row's cache
  base is `slot * local`; under DCP `local` is the shard's row count, so the tables are already right.
- **Prompt fills.** A prompt fills through `Runner.prefillChunk` on its slot's view with the one-stream kernels, which
  is the path the 1M shape serves with.
- **Prompt reuse.** With several streams a kept state stays in its slot's rows; `rowViews` already counts a rank's
  share. Only the slot-to-slot copy counted wrong (above).
- **Graphs.** Windows are captured per shape as before; the tables are static buffers.
- **Drafters.** DSpark and DFlash2 read the windows' taps and keep their own per-stream rings.

## Layout

A stream's shard on a rank is `local = ceil(capacity / dcp) + 1` rows of each cache (`capacity = context + k + 1`).
The slots sit back to back. Stream `s`, position `p`:

```
rank  = p % dcp
row   = s * local + p / dcp          (BASE[r] = s * local, the window's int32 table, as at DCP 1)
```

The kernel changes (`tools/glm53/patch_fused_dcp_rows.py`, applied to a copy of the Python source before the sweep):

| Kernel | Change |
|---|---|
| `_kv_write`, `_ik_write` with `ROWS` | store only on the owner (`POS[r] % DCP == RANK`), at `BASE[r] + POS[r] // DCP`. At `DCP` 1 the address is unchanged, so packed variants stay valid. |
| `_index_scores` | none: `ROWS` moves the row's position and index-key base, `DCP` maps the slot to its global position. New variants only. |
| `_attn_dcp_rows` (new) | `_attn_dcp` with `p = POS[r]` and the cache read from `LC + BASE[r]` rows. A new function, so no variant of `_attn_dcp` in an older pack can answer its launch. |

## What a window exchanges

Per layer, for a window of R rows over S streams (R = the streams' rows together; G = 4 ranks, H = 16 heads a rank):

| Exchange | Bytes a rank sends | 3 rows | 16 rows | 32 rows |
|---|---|---|---|---|
| queries (`qpack`) | R x H x 576 x 2 | 55 KB | 295 KB | 590 KB |
| partials (`osend`) | G x R x H x 512 x 2 | 197 KB | 1.05 MB | 2.1 MB |
| log-sum-exps | G x R x H x 4 | 0.8 KB | 4 KB | 8 KB |
| candidates (21 indexer layers) | R x 2048 x 8 | 49 KB | 262 KB | 524 KB |
| the two residual sums | R x 6144 x 4 each | 74 KB | 393 KB | 786 KB |

The RoCE one-shot takes a gather only while `world x bytes <= 1 MiB` (`cuda_roce.max_bytes`). The partials pass it
up to 4 rows, the queries up to 14, the candidates up to 16; past that the gather is NCCL's. The 1M shape already
runs its 8-row DSpark windows this way. Windows of several streams are wider, so more of their exchanges are NCCL's.
Raising the one-shot's limit is the obvious follow-up; it is not part of this change.

## Memory

A token of context costs 96,640 bytes over all ranks: 78 latent rows of 1,152 bytes, 21 index keys of 256 bytes, and
the MTP layer's row and key (`cuda_dcp.tokenBytes`). A rank holds

```
cache bytes = 96,640 x (ceil((context + k + 1) / dcp) + 1) x streams
```

| Shape | DCP | Cache a rank | Left on a node |
|---|---|---|---|
| 32K x 4 | 1 | 11.8 GiB | 9-10 GB measured (dense MTP copy on) |
| 64K x 2 | 1 | 11.8 GiB | 12-14 GB measured |
| 1M x 1 | 4 | 22.5 GiB | 8-10 GB measured (dense MTP copy off, 5 GB; kept prefixes capped at 4 GB) |
| 500K x 2 | 4 | 22.5 GiB | as 1M x 1, less the second stream's drafter ring and wider decode buffers (estimate: under 1 GB) |
| 250K x 4 | 4 | 22.5 GiB | as 1M x 1, less three more streams' of the same (estimate: 1-2 GB) |
| 140K x 4 | 1 | 50.4 GiB | does not fit: the shape that took four Sparks down |

The last column of the new shapes is an estimate. With several streams the kept prompt states live in the slots'
rows, so the 4 GB prefix store of the 1M shape is not allocated; that offsets part of the per-stream cost.

Boot checks (every rank, before anything allocates the caches):

1. `context x streams <= 1,048,576` tokens (`cuda_dcp.streams_tokens`). Nothing larger than the 1M shape has been
   held on a node.
2. The slots' whole allocation (index keys and the MTP layer included) plus a reserve must fit what the device
   reports free. The reserve is 8 GiB (`TF_GLM53_DCP_STREAMS_RESERVE_GB`) and is never halved, unlike the one-stream
   check's. A shape that fails is refused with the context that would fit.

Use `TF_GLM53_MTP_DENSE=0` with these shapes, as the 1M preset does.

## Speed: what the 1M shape pays, and what several streams inherit

Measured on the 1M shape against 32K x 4: prompts are read about 2.5 times slower, code decode is 44.7 against
56.3 tok/s, prose 35.0 against 38.3.

| Cost | Where it goes | Inherited by several streams under DCP |
|---|---|---|
| Prompt chunks of 1,024 rows instead of 8,192, no sequence-parallel chunk | every rank attends all 64 heads over its own keys, and the chunk's partials cross the ranks by NCCL all-to-all | yes, per stream; and prompts fill one chunk a round between decode rounds, so a long prompt slows the streams that are decoding |
| Three more gathers a layer in decode (queries, partials, log-sum-exps), one more on indexer layers | 78 layers x 3 + 21 gathers a window, the wide ones through NCCL | yes, and the windows are wider (see the table above) |
| No dense MTP copy, no MTP index reuse | drafts cost more | yes (reuse is off with several streams at DCP 1 too) |

Expectation, not a measurement: one stream alone in a 250K x 4 server decodes at about the 1M shape's rate (33-35
prose, about 45 code). Aggregate throughput with four streams should rise as it does at 32K x 4, but by less,
because the partials' gather grows with rows and leaves the one-shot path. I would not be surprised by an aggregate
of 1.3-2 times one stream, against 2-2.5 times at 32K x 4.

## Kernel coverage

`tf-glm53-generate --mode list-variants --dcp 4 --parallel N` names the launches. Against the 1M kernel set of
2026-10-10 (no drafter kernels), `--context 500000 --parallel 2` and `--context 250000 --parallel 4` each need 243
variants, 52 missing:

| Missing | Count | In the wide pack already |
|---|---|---|
| `_attn_dcp_rows` | 12 | no (new) |
| `_kv_write`, `_ik_write` with `ROWS` and `DCP` 4 | 4 + 4 | no (new) |
| `_index_scores` with `ROWS`, `DCP` 4, `PACK` | 4 | no (new) |
| `_absorb`, `_iq_rope` with `ROWS` | 17 + 1 | expected yes (64K x 2 uses them) |
| DSpark's kernels | 10 | expected yes |

So 24 new variants for certain. `~/mlx-serve-work/dcps/aot.sh` lists against the wide pack on a node, patches a
copy of the Python source, compiles what is missing (`sweep_aot.py`, no launch, one GPU) and merges the result into a
new pack `aot-dcps`. Estimate: 5-10 minutes.

## Settings

| Setting | Meaning |
|---|---|
| `TF_GLM53_DCP_STREAMS=1` | every rank: `--parallel N > 1` with DCP on is served. Unset: refused as before. |
| `TF_GLM53_DCP_STREAMS_RESERVE_GB` | GiB that must stay free after the slots' caches (default 8) |
| `TF_GLM53_DCP` | unchanged: 0 auto (on past 200,000 tokens of context), 1, or 4 |

Defaults are unchanged: without `TF_GLM53_DCP_STREAMS` the engine starts and refuses exactly as before, and at
DCP 1 the multi-stream launches name the same variants they always did.

## Gates (GPU, four Sparks)

`~/mlx-serve-work/dcps/window.sh`, in order:

1. **prep** - the branch on the nodes, built in the image.
2. **aot** - the missing variants compiled, `list-variants` reporting nothing missing for each shape.
3. **cli** - `tf-glm53-generate --mode 4b --dcp 4` at TP4: prompts of unequal length, some past 2,048 tokens (where
   selection starts), run together, each alone through the concurrent decoder, and each through the one-stream path
   (the 1M shape's kernels). Every stream's tokens must be the same in all three. This is the correctness gate.
4. **served** - 250K x 2, 500K x 2, 250K x 4 behind the HTTP server, each with a memory guard at 6 GB: the same
   prompts' replies against a one-stream DCP boot, a long-prompt needle on every stream at once, per-stream and
   aggregate tok/s for prose and code with thinking on, prefill tok/s.

## Not done

- A stream's prompt past 200,000 tokens is admitted like any other; `cuda_engine`'s one-stream auto point is not a
  per-stream limit any more, and the README sentence that says so needs changing when this ships.
- DFlash2 and MTP drafting are wired the same way as DSpark but only DSpark is in the gate.
- DCP 2 (two ranks a shard) is still refused: `choose` takes 1 or the world.
- The quantized latent cache (`_kv_write_q`) takes the same patch and is launched by nothing here.
