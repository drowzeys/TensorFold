# GLM-5.3 speed settings

Every setting here is off by default: with none set the engine launches what 1.0.7's does. The exchange trace follows
the one in Jay Leaton's DeepSeek-V4.1-Flash family of TensorFold; the rest comes from our own comm probe. What was
measured on four DGX Sparks is in "Measured" below.

| Setting | What it does | Bits |
|---|---|---|
| `TF_GLM53_ROCE_BF16=1` | Decode exchanges send bf16 partials: half the bytes. | Changes bits |
| `TF_GLM53_ROCE_HEALTH_LOCAL=1` | The per-window health check reads this rank's own word: no NCCL all-gather a window. A failure surfaces one window later. | Same |
| `TF_GLM53_L2PF_AFTER=1` | The L2 prefetch sites are queued after their exchange, not beside it. | Same |
| `TF_GLM53_ROCE_TRACE=65536` + `TF_GLM53_ROCE_TRACE_FILE=path` | Every exchange stamped on the GPU's timer; each rank appends `path.r<rank>`. | Same |
| `TF_GLM53_HEAD_Q=<file>\|pack` | `lm_head` served as an EXL3 linear (3, 4, 5, 6 or 8 bits) instead of bf16 rows: fewer head bytes a window. | Lossy |
| `TF_GLM53_KVB_Q=1` | A pack whose `kv_b_proj` is an EXL3 linear loads: decoded to the bf16 blocks at load. A smaller pack, the same bytes a token. | As the pack |

The ranks compare the two wire settings, the quantized head's bytes and the kv_b setting before serving.

## Measured (2026-10-12, four DGX Sparks, GLM-5.3 EXL3 2.75 bpw, 32,768 tokens x 4 streams, DSpark drafter, thinking on)

One setting a boot; the same build with nothing set gave 37.8 prose / 55.9 code tok/s first and 37.9 / 56.1 last, and
scores KL 0.175612 against the unquantized checkpoint (`docs/glm53-score.md`), as the build before these settings did.

| Setting | Prose / code tok/s | Replies against nothing set | Quality |
|---|---|---|---|
| `TF_GLM53_ROCE_BF16=1` | 39.2 / 56.6 | 1 of 10 identical | KL on tool-call rows 0.5747 -> 0.5773 (decode path), all rows 0.17737 -> 0.17770: left off |
| `TF_GLM53_L2PF_AFTER=1` | 36.4 / 54.0 | 10 of 10 | slower: left off |
| `TF_GLM53_ROCE_HEALTH_LOCAL=1` | 38.4 / 56.1 | 10 of 10 | none |
| `TF_GLM53_HEAD_Q` (6 bits) | 39.2 / 57.5 | 8 of 10 | KL 0.175612 -> 0.175724; 0.79 ms a head launch against 2.16 ms |

The published image `2026-10-12` sets the last two.

## The exchange trace

A decode window is about 156 exchanges; at each one every rank waits for the last rank to arrive. The trace shows
which rank that is and where in the window.

```sh
TF_GLM53_ROCE_TRACE=65536 TF_GLM53_ROCE_TRACE_FILE=/work/log/roce-trace   # on every rank: each appends <file>.r<rank>
python3 tools/glm53/roce_trace.py roce-trace.r0 roce-trace.r1 roce-trace.r2 roce-trace.r3
python3 tools/glm53/roce_trace.py --selftest            # no GPU
```

It prints, for each rank, how often it arrived last, how long it stood ready, and its compute between exchanges;
then the positions in a window where the ranks' compute differs most. The ranks' timers are not synchronized, so
only one rank's own intervals are compared. The file is written between windows once half the ring is new, and when
the engine shuts down cleanly.

What to try once the trace names a cause: `TF_GLM53_L2PF_AFTER=1` if the exchange's stage and copy are fat beside
the prefetch, `TF_GLM53_ROCE_HEALTH_LOCAL=1` for the per-window NCCL all-gather.

## A quantized head (`families/glm_moe_dsa/headq.zig`)

Every window ends in the head: this rank's 38,720 rows of `lm_head`, 476 MB of bf16, measured at 2.16 ms a launch on
rank 0 (2026-10-12). `TF_GLM53_HEAD_Q` serves an EXL3 `lm_head` instead, through the launches every dense EXL3
linear takes (`rot_in`, `linear_kernel`), and at 6 bits a rank's part is 179 MB.

The vocabulary stays split as it is (rank r owns tokens r V / world ..). An EXL3 linear can only be cut along whole
128-wide Hadamard blocks and a share is 302.5 of them, so a rank holds the 303 blocks that cover its share, one of
them also held by a neighbour, and copies its share out of the launch's result (`glm53_copy_rows`, one more small
launch). A token's logit comes from its own block's tiles and scales alone, so both holders of a shared block
compute the same bits for it (host-tested on the CPU replay).

| What | State |
|---|---|
| 6-bit `linear_kernel` / `unpack_kernel` (K2 12) | The template's own code, instantiated and compiled into `glm53_exl3_linear.fatbin`. Its device logits match the CPU replay on all four ranks (`--score-head-check`). |
| Decode windows, a prompt's last row, the MTP head, sampled candidates, the scorer's every row | One function, `cuda_forward.headLogits`: all of them read the same logits buffer as before. |
| The drafters' 4-bit head, the reduced draft head | Still made from the bf16 `lm_head.weight` (drafts only propose; the quantized head verifies). The checkpoint must still hold it. |
| Tile tables, Triton packs | Unchanged: the head is in no tile table (it takes `x3linear.plan`'s tiles: 1 split, 8 warps) and its launches are fatbin kernels. `_router_part` over the vocabulary share is no longer launched. |
| Split-K heads (a rank's part under 64 blocks) | Refused at load (`BadSplit`): no partials scratch is wired. GLM-5.3's is 303 blocks. |

What a file must hold (the 6-bit file published beside the 2.75 bpw pack does): `lm_head.trellis` int16
[hidden / 16, vocab / 16, 16 x bits], `lm_head.suh` fp16 [hidden], `lm_head.svh` fp16 [vocab], `lm_head.mul1`.
The value is the file's path (the same file on every rank: the ranks compare its scales and its trellis' ends), a
directory holding `lm_head.safetensors`, or `pack` when the checkpoint itself holds those tensors. With `pack` and
a drafter the checkpoint must hold `lm_head.weight` as well.

```sh
python3 tools/glm53/head_q_ref.py --selftest                          # the numpy reference, no GPU
# on every rank, with the engine's usual flags: the gate, the time and the KL
tf-glm53-generate --mode score --head-q /model/lm_head-6bit/lm_head-6bit.safetensors --score-head-check 3 --score-time 200 ...
# served: TF_GLM53_HEAD_Q=/model/lm_head-6bit/lm_head-6bit.safetensors on every rank
```

`--score-head-check N` runs N fixed rows through the device head and through `headq.replay` on the CPU and prints a
`HEADQ` line a rank: it fails when a logit differs by more than 1e-4 of the largest, or a row's first maximum
moves. The replay agrees with the numpy reference to 2e-5 on the host (rows of four shapes, 5, 6 and 8 bits, a
share that starts inside a block); the device's tile steps are `mma` sums whose inner order the replay cannot know,
so the device is not expected to match it bit for bit.

On the host (2026-10-12, no GPU): `tf-glm53-generate --mode head-q-info` reads the first requant's file as 6.0
bits, mul1, 6144 x 154880, 170 MiB a rank, one K split; rank 1's launch (the share that starts 64 outputs into a
block) replayed on the CPU for three unit-normal rows gives the unquantized `lm_head`'s logits of those tokens to
2.2 % (rms of the difference over rms of the logits, cosine 0.9998) and the same first maximum on all three. That
is a weight-space comparison on random inputs, not a KL on hidden states.

Measured on four Sparks: 0.77 to 0.82 ms a launch at 6 bits against 2.16 ms, and KL 0.175724 against 0.175612.

## kv_b

`kv_b_proj` is bf16 in every pack: 29 MB a layer over the ranks, 2.3 GB a token over the model, more than the head.
The attention does not apply it as a linear: the key rows are absorbed into the query (`q_nope @ wk`, the transposed
product, one 192 x 512 block a head) and the value rows expand the attended latent (`ol @ wv^T`, a different input
for each head). Neither is a launch `linear_kernel` can make, and a head's 448 outputs are 3.5 Hadamard blocks, so
a block is shared by two heads with different inputs. Reading kv_b quantized in a window needs two new kernels (a
transposed walk for the absorb, a per-head forward for the expand); they are not written.

What is built: `TF_GLM53_KVB_Q=1` loads a pack whose kv_b is an EXL3 linear by decoding it on the host to the dense
matrix (`headq.dense`: diag(suh) H W_q H diag(svh) in float64, checked against the reference decoder), rounding to
bf16 and cutting it into the same `wk` / `wv` blocks. The pack is smaller (2.3 GB to 0.7 GB at 5 bits); a token
reads the same bf16 bytes and the attention launches nothing new. It makes a quantized kv_b's quality measurable
with the scorer before the kernels are worth writing. The decode walks 3.7 M values bit by bit for each layer and rank on
the host at load (not timed).

The converter leaves kv_b in bf16 today. For this setting it must emit, for every layer and for the MTP layer,
`model.layers.<i>.self_attn.kv_b_proj.{trellis,suh,svh,mul1}` in place of `.weight`: an ordinary EXL3 linear with
`kv_lora_rank` (512) inputs and `num_attention_heads x (qk_nope_head_dim + v_head_dim)` (28,672) outputs in the
checkpoint's row order, so `trellis` is int16 [32, 1792, 16 x bits], `suh` fp16 [512], `svh` fp16 [28672] (fp16
scales, not packed sign words), codebook `mul1`, 3, 4, 5, 6 or 8 bits. The ranks cut it by outputs (a rank's 16
heads are 56 whole blocks).

## Not built

A fused dense prompt GEMM (EXL3 decode and GEMM in one kernel): expected 0 to 3 % of prefill at our 8192-row
chunks, and it cannot be developed without a GPU to iterate on.
