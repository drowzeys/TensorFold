# GLM-5.3 speed settings (branch glm53-speed)

Every setting here is off by default: with none set the engine launches what 1.0.7's does. **None has run on a GPU
yet**; the rules and file formats are host-tested. The ideas come from Jay Leaton's DeepSeek-V4.1-Flash family of
TensorFold (routed-expert pruning, the exchange trace) and from our own comm probe.

| Setting | What it does | Bits |
|---|---|---|
| `TF_GLM53_EXPERT_TOPP=0.85` | A row keeps the routed experts that reach this share of its routed weight, at least `TF_GLM53_EXPERT_MIN_K` (3). Fewer expert bytes a decode window. | Lossy |
| `TF_GLM53_EXPERT_RENORM=orig\|kept` | Kept weights left as routed (`orig`, default: the routed output shrinks by the dropped share) or rescaled to the full sum (`kept`). | |
| `TF_GLM53_EXPERT_PRUNE_PROMPT=1` | Prompt chunks prune too. Default 0: only windows the decode kernels run (generated tokens, and prompt chunks of at most 16 rows). | Lossy |
| `TF_GLM53_ROCE_BF16=1` | Decode exchanges send bf16 partials: half the bytes. | Changes bits |
| `TF_GLM53_ROCE_HEALTH_LOCAL=1` | The per-window health check reads this rank's own word: no NCCL all-gather a window. A failure surfaces one window later. | Same |
| `TF_GLM53_L2PF_AFTER=1` | The L2 prefetch sites are queued after their exchange, not beside it. | Same |
| `TF_GLM53_ROCE_TRACE=65536` + `TF_GLM53_ROCE_TRACE_FILE=path` | Every exchange stamped on the GPU's timer; each rank appends `path.r<rank>`. | Same |

The ranks compare the pruning rule and the two wire settings before serving. A learned prompt state made with
pruning is never reused without it.

## Pruning

`prune.zig` holds the rule and its host replay; `glm53_expert_prune` (glm53_decode.cu) runs it in place on the
router's picks and weights, one thread a row, right after the router's top-k. A dropped slot gets weight 0 and the
pick 256 (the expert count): the grouped decode kernels and the prompt experts both skip a pick outside [0, 256), so
none of its bytes are read. The MTP layer is never pruned.

Measured on the router's weights of 137,570 positions (`tools/glm53/prune_ref.py stats`, 2026-10-12):

| TOPP | Experts kept of 8 | Routed weight dropped |
|---|---|---|
| 0.80 | 5.69 | 15.8 % |
| 0.85 | 6.33 | 10.5 % |
| 0.90 | 6.85 | 6.9 % |

Quality is measured with the scorer (`docs/glm53-score.md`): `--expert-topp 0.85` on `tf-glm53-generate --mode score`.
`--score-rows 16` scores the decode kernels' windows (always pruned); `--score-rows 0` scores prompt chunks, pruned
only with `--expert-prune-prompt 1`.

```sh
python3 tools/glm53/prune_ref.py cases zig/src/families/glm_moe_dsa/prune_cases.bin --route ROUTE.bin   # the fixture
python3 tools/glm53/prune_ref.py stats ROUTE.bin
```

## The exchange trace

A decode window is about 156 exchanges; at each one every rank waits for the last rank to arrive. The trace shows
which rank that is and where in the window.

```sh
ROCE_TRACE=65536 ~/mlx-serve-work/cluster.sh up        # each rank appends $W/log/roce-trace.r<rank>
python3 tools/glm53/roce_trace.py roce-trace.r0 roce-trace.r1 roce-trace.r2 roce-trace.r3
python3 tools/glm53/roce_trace.py --selftest            # no GPU
```

It prints, for each rank, how often it arrived last, how long it stood ready, and its compute between exchanges;
then the positions in a window where the ranks' compute differs most. The ranks' timers are not synchronized, so
only one rank's own intervals are compared. The file is written between windows once half the ring is new, and when
the engine shuts down cleanly.

What to try once the trace names a cause: `TF_GLM53_L2PF_AFTER=1` if the exchange's stage and copy are fat beside
the prefetch, `TF_GLM53_ROCE_HEALTH_LOCAL=1` for the per-window NCCL all-gather, Living Weights' spread placement
if rank 0 is last at the FFN exchanges.

## Not built

A fused dense prompt GEMM (EXL3 decode and GEMM in one kernel): expected 0 to 3 % of prefill at our 8192-row
chunks, and it cannot be developed without a GPU to iterate on.
