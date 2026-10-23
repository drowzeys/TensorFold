# Quantized checkpoints

TensorFold reads the precision stored in a checkpoint, including its per-layer settings.
Serve a local or Hugging Face checkpoint through the usual CLI:

```bash
tensorfold info ./model-8bit
tensorfold serve ./model-8bit
```

This build extends the Qwen3.5/3.8 dense family to MLX affine weights at 2, 3, 4, 5, 6 and 8 bits, with group sizes 32, 64 and 128.
It keeps packed words, scales and biases at their stored precision, including mixed layer formats.
It does not convert a checkpoint during loading.
Choose an 8-bit checkpoint made from the original model when you want its 8-bit quality.

## Backend paths

The table, `--lane-kernels` and the CUDA path below describe the Python engine, frozen at 0.6.6 and available at the
[`v0.6.6` tag](https://github.com/ashhart/TensorFold/tree/v0.6.6). The native binary refuses `--lane-kernels`. Its
Qwen3.8-27B family reads the same affine formats, per module (`zig/src/families/qwen27/affine.zig`), and admits each
against the projection kernels the chip runs (`zig/src/families/qwen27/widths.zig`).

| Path | Formats |
| --- | --- |
| Qwen dense, M5 native lane kernels | Affine 2/3/4/5/6/8-bit, group 64; affine 4-bit also group 32 |
| Qwen dense, packed row kernels on Apple Silicon | Affine 2/3/4/5/6/8-bit, groups 32/64/128 |
| Qwen dense, CUDA | Affine 2/3/4/5/6/8-bit, groups 32/64/128 |
| Nemotron | Existing affine 4-bit recipe; broader expert formats are not enabled here |
| Flash Next | Existing uniform affine 4-bit/group-32 recipe |
| GLM and EXL3 | Existing family recipes and separate EXL3 integration |

With `--lane-kernels auto`, M5 uses its native kernels when they support the checkpoint's formats and otherwise uses the packed row decoder.
M1 through M4 use the packed row decoder.
`--lane-kernels on` requires an M5-generation GPU and formats supported by its native kernels; an incompatible format is refused with guidance.
The language path retains BF16 activation requirements.
The new Metal affine projection reader accepts BF16, FP16 or FP32 affine scales and biases with matching metadata dtypes.

All paths still verify draft rows through the lane engine.
The existing specialized 4-bit routes remain available; the general readers make different formats executable without expanding the entire model into dense weights.
Their performance can differ substantially from the specialized routes.

## Mixed precision

The loader reads `quantization` before the legacy `quantization_config`, matching the current MLX loader.
A nonempty per-module mapping uses that mapping's own defaults, so `{"bits": 8}` means affine 8-bit with group 64 even when the global group is 128.
Boolean `true` inherits global settings; `false` and an empty mapping disable quantization for that module where the family supports its resulting unquantized projection.
Conflicting aliases and packed shapes that disagree with the declared bit width or group size are refused.

Fused stacks combine projections only when bit width, group size, input width and metadata dtypes agree.
Otherwise the model computes those projections separately with their declared formats.
In the Python 0.6.6 engine, CUDA tensor-parallel splits retain complete groups and packed-word boundaries, and memory
admission counts the actual packed precision.
Changing a checkpoint, precision or kernel implementation also changes the conditions under which cached states remain valid.

## Scope and qualification

These additions cover MLX affine safetensors checkpoints for the dense Qwen family.
GGUF, AWQ, GPTQ, NVFP, MXFP and arbitrary 8-bit MoE expert formats require their own readers and kernels.
Unsupported formats remain explicit errors.

The new paths are experimental until their hardware qualification is complete.
Each backend must pass packed-value checks, one-row versus batched output equality, full-model drafted versus serial equality, prefix resume, concurrent-stream equality and memory admission before a release claim.
In the Python 0.6.6 engine, the Metal kernel has a dedicated hardware test in
[`tests/test_affine_rows_metal.py`](https://github.com/ashhart/TensorFold/blob/v0.6.6/tests/test_affine_rows_metal.py);
model and GPU tests must run under the machine's existing resource controls.
No speed or quality improvement is implied by a higher bit width.

## Native safetensors metadata

The shared Zig safetensors index retains up to five dimensions, including temporal patch convolutions
such as BF16 `[1024, 3, 2, 14, 14]`. It still rejects ranks above five, invalid dimensions,
overflowing byte counts, and offsets that disagree with tensor size or exceed the data region.
Namespace selection remains available to text loaders that do not interpret a vision tower's layouts.
Reading this metadata does not qualify a vision encoder, checkpoint or device.

## Drama 3 routed experts in NVFP4

`tf-drama-quant` writes a new Drama 3 checkpoint whose 256 routed experts per layer are NVFP4: e2m1 codes, an e4m3
scale every 16 inputs and one fp32 global per projection, under the ModelOpt names the NVFP4 expert kernels read.
It runs on the CPU and is a lossy repack: the FP8 checkpoint it reads stays the faithful one. The same run can also
store dense projections in FP8 ([below](#drama-3-dense-projections-in-fp8)).

```sh
zig build drama-quant
zig-out/bin/tf-drama-quant drama-3/ drama-3-nvfp4/ [--stats stats.safetensors] [--keep-fp8 0,39] [--encoded gptq/]
```

- **Source.** The source checkpoint is only read, and its index may name shards only as plain `NAME.safetensors` files
  inside it. A symlink anywhere in the source is refused. The destination must not exist and must not lie inside the
  source. Index paths, requested layer selections, encoded tensors, statistics values and ancillary file kinds are
  checked before creating the staging directory. Source tensor validation and I/O can still fail during conversion;
  failed runs do not publish DST.
- **Publishing.** The checkpoint is written into `DST.partial` (refused if it exists), which becomes `DST` only once the
  shards, index, copied files and provenance are complete; a failed run removes it, so no unfinished `DST` remains.
- **Tensors.** In each converted layer, every `mlp.experts.E.{gate,up,down}_proj.weight` (F8_E4M3 with a 128 × 128
  `weight_scale_inv`) becomes `weight` (U8 [n, k/2], even inputs in the low nibble), `weight_scale` (F8_E4M3 [n, k/16])
  and `weight_scale_2` (F32 scalar) in its place; `weight_scale_inv` and any `input_scale` leave, and any other tensor
  of a converted projection is refused. Every other tensor keeps its bytes, shard and order. The index is written anew
  and every other file of the source (config.json, codec.pth, the tokenizer, nested directories) is copied at its
  relative path.
- **Encoding.** The FP8 value times its scale_inv, in f64, is the target. The global is `amax / (6 × 448)` in f32. Each
  16-input block tries all 126 positive finite e4m3 scales; at each, every weight takes the nearest e2m1 value (ties
  to the even one, saturating at ±6), and the block keeps the scale of least `Σ h (W − Q)²`, the smaller code on a tie.
  The output is the same for any thread count. Non-finite weights and non-positive scale_inv are refused.
- **Importance.** Without `--stats`, h = 1. With `tf-drama-probe`'s statistics, the per-class sums of squares are
  pooled; gate and up weigh input c by `(S_e,c + 64 m_c) / (n_e + 64)` with m the layer's mean, down by
  `(S_e,c + 64 m̄_e) / (n_e + 64)` with m̄_e the expert's mean. A layer no pair reached uses h = 1; within an observed
  layer, an expert no pair reached uses h = 1 for down, while its gate and up inherit the layer mean. A count or sum
  that is negative or not finite is refused, never read as no observation.
- **`--keep-fp8`.** The listed layers' experts stay FP8, byte for byte.
- **`--encoded`.** A directory of `layer-LL.safetensors` files, each holding exactly one layer's C1 tensors under their
  checkpoint names (as an external encoder such as a GPTQ tool writes them), supplies those layers byte for byte instead
  of encoding them. Each file is checked first: every expert and projection present at the source's shapes, nothing
  else, scale bytes finite and non-negative, globals finite and positive. A layer both encoded and kept FP8 is refused.
  Their errors are still measured against the FP8 source, so both kinds of layer compare in one table.
- **Provenance.** `quant-provenance.json` records under `experts` (`mode` `nvfp4`) the source, config.json's SHA-256,
  the tool's commit, the statistics file and its class mixture, the layers converted (each `rtn` or `encoded`, with the
  file's SHA-256 and metadata), the wall time and, per projection, amax, global, `rel` (‖W − Q‖ / ‖W‖) and `err_h`
  (the same weighed by h: an expert-input error proxy, not the layer's output error), with their mean and maximum per
  layer and projection.
- **Measured.** On a GB10 (20 CPU cores, sharing them with another job at nice 19) the 40-layer conversion without
  statistics took 9 min 10 s at 7.2 GiB peak resident memory. The shards went from 36.88 to 23.76 GiB; `rel` averaged
  0.0811 (0.0517 to 0.0824 per projection).

## Drama 3 dense projections in FP8

`tf-drama-quant --dense-fp8` stores the large BF16 projections of Drama 3's backbone and MTP tower as FP8 for the
fp8g lane (W8A16): e4m3 codes with one f32 scale per output row and 64 inputs. Activations stay BF16. Like the NVFP4
experts this is a lossy, experimental checkpoint beside the faithful one; the engine serves it
([backbone](drama-voices.md#fp8-dense-projections), [tower](drama-voices.md#fp8-tower-projections)).

```sh
zig-out/bin/tf-drama-quant drama-3-nvfp4/ drama-3-fp8/ --experts keep --dense-fp8 all [--keep-bf16 shared_in,b:0,3]
```

- **Classes.** Each class is one fused group of a layer, converted or kept whole: `gdn_in` (`linear_attn.in_proj_qkv`
  and `in_proj_z`), `gdn_out` (`linear_attn.out_proj`), `attn_in` (`self_attn.q_proj`, `k_proj`, `v_proj`),
  `attn_out` (`self_attn.o_proj`), `shared_in` (`mlp.shared_expert.gate_proj` and `up_proj`), `shared_out`
  (`mlp.shared_expert.down_proj`), and in the tower `tower_qkv` (`attention.wqkv`), `tower_o` (`attention.wo`),
  `tower_gate_up` (`feed_forward.w1` and `w3`) and `tower_down` (`feed_forward.w2`). Every other tensor keeps its bytes:
  `in_proj_b`/`in_proj_a`, `conv1d`, `A_log`, `dt_bias`, norms, the router, `shared_expert_gate`, embeddings,
  `lm_head`, and the tower's `eh_proj`, norms, embeddings and heads.
- **Selectors.** `--dense-fp8` takes classes or `all`. `--keep-bf16` then removes whole groups: a class, backbone
  layers `b:L[,L]` or tower layers `t:L[,L]`, where t:0 to 3 are `audio_decoder_mtp_shared.layers.0-3` and t:4 to 12
  `audio_decoder_mtp.layers.0-8`; a bare number belongs to the namespace before it, and every layer is plain decimal
  digits (no sign, `_` or leading zero). `--dense-fp8 all --keep-bf16 shared_in,b:0,3` is valid. Unknown classes,
  tensor names, unnamespaced or malformed layers, layers the source lacks, and `--keep-bf16` without `--dense-fp8`
  are refused.
- **Experts.** `--experts nvfp4`, the default, converts the routed experts as above and needs them in FP8. `--experts
  keep` copies every expert tensor and companion byte for byte, FP8 or NVFP4, and refuses `--keep-fp8` and
  `--encoded`; a Phase 1 checkpoint converts with `--experts keep --dense-fp8 all`.
- **Tensors.** Each converted `X.weight`, BF16 [n, k] with n a multiple of 128 and k of 64, becomes `X.weight` F8_E4M3
  [n, k] followed by `X.weight_scale_inv` F32 [n, k/64] in its place. A selected tensor of another dtype or shape, or
  one that already has a `weight_scale_inv`, is refused before staging. The source, publishing and index rules are
  those above.
- **Encoding.** Per row and 64 inputs, the bf16 values are widened to f64 (a non-finite one is refused) and
  scale_inv = amax / 448, rounded once to f32 (1.0 for an all-zero group). Each code is w / scale_inv rounded directly
  to the nearest finite e4m3fn value, exact ties to even, saturating at ±448 (never 0x7F or 0xFF); zero is 0x00.
  There is no scale search. The output is the same for any thread count.
- **Statistics.** With `--stats`, `shared_in`'s `err_h` weighs input c by the layer's routed gate/up sums of squares
  summed over its experts and classes: the shared expert reads the same row. With `--experts keep` they serve only
  this. Statistics that no operation reads (no expert layer encoded, no `shared_in` converted) are accepted, as
  before, and recorded as not consumed.
- **Provenance.** `quant-provenance.json` records `experts` (`nvfp4` as above, or `keep`: nothing encoded),
  `dense_fp8`, `stats` (the `--stats` file's path and SHA-256 and whether any operation read it; null without
  `--stats`), `stats_consumed_by` (`experts`, `dense_fp8` or neither), and `source_provenance`: the source's own
  `quant-provenance.json`, or a Phase 1 `nvfp4-provenance.json`, as file name, SHA-256 and exact text, null when
  there is none. `dense_fp8` holds the source path, index and config.json SHA-256, both selectors, the resolved
  classes, the scale rule, the commit, the wall time and, per converted tensor, class, layer, `rel`, `worst_row_rel`,
  `max_abs` and, for `shared_in` with statistics, `err_h`. It is null only when `--dense-fp8` was not given; a
  selection that `--keep-bf16` empties keeps its record with an empty tensor list. A source's `nvfp4-provenance.json`
  is also copied as it is.

## Drama 3 compact routed experts

`tf-drama-quant --compact-experts MASK` writes only the routed experts a mask keeps, so a pruned expert set stops
costing memory; the engine serves it ([docs](drama-voices.md#compact-experts)). MASK is an `--expert-mask` file
([format](drama-voices.md#expert-mask)), `{"layer": [kept expert ids]}`: the same file that tries the set by routing
alone tries it here in storage.

```sh
zig-out/bin/tf-drama-quant drama-3/ drama-3-small/ --experts keep --compact-experts kept.json
zig-out/bin/tf-drama-quant drama-3-nvfp4/ drama-3-nvfp4-small/ --experts keep --compact-experts kept.json
```

- **Tensors.** Every tensor of an expert the mask drops, companions included, is left out; a held expert keeps its
  original name, bytes and order, so `experts.200` stays expert 200. A layer the mask leaves out holds all of its
  experts. Compaction comes first: `--experts nvfp4` then encodes only the held FP8 experts, and since each projection
  encodes on its own, they come out as the full conversion's bytes. `--experts keep` copies held FP8 or NVFP4 experts
  byte for byte. Router rows are not touched: routing keeps the full layer's logits.
- **Map.** The index's `metadata.drama_experts` maps every layer that has experts to its held ids, ascending:
  `{"0": [0, 1, ..., 255], "3": [2, 5, ...], ...}`. A later run that does not compact carries the map on unchanged;
  compacting a checkpoint that already has one is refused.
- **Refused.** A mask the server would refuse (`tf-drama-quant` prints why), a layer the mask prunes that has no
  experts in the source, and an id past the layer's last expert, all before anything is written.
- **Provenance.** `quant-provenance.json`'s `compact_experts` holds the mask's path and SHA-256, the experts held and
  the source's total; null without `--compact-experts`.

## Drama 3 routed experts by GPTQ

`tools/zig/drama_nvfp4_gptq.py` encodes Drama 3's 256 routed experts per layer to NVFP4 with GPTQ error feedback,
from the rows `tf-drama-probe --rows` dumps. It writes one `layer-LL.safetensors` per layer holding exactly that
layer's NVFP4 expert tensors, under the names, dtypes and shapes `tf-drama-quant` writes. The files are not a
checkpoint: `tf-drama-quant --encoded DIR` assembles them into one. Like `tf-drama-quant`, this is a lossy repack.

```sh
venv/bin/python tools/zig/drama_nvfp4_gptq.py drama-3/ rows/ gptq/ [--layers 0,17,39]
```

- **Needs.** torch with CUDA, Triton and safetensors (the pilot venv: torch 2.13 cu130, Triton 3.7.1, safetensors
  0.9). Without a GPU the same kernels run in Triton's interpreter, which is fast enough only for the tests.
- **Rows.** `rows.json` (`drama-probe-rows/1`) must carry the source's config.json SHA-256 and declare R rows of
  `class.u8` [R], `layer-LL.x.bf16` [R][2048] and `layer-LL.picks.i32` [R][8]. File lengths are checked against it,
  as are finite inputs, classes 0 to 2, and picks of 8 distinct experts from 0 to 255. Classes are pooled by row count.
  Routing weights (`layer-LL.w.f32`) do not enter: the Hessians are unweighted second moments of the expert inputs.
- **Hessians.** gate and up share H_e = Σ x xᵀ over the rows routed to expert e, shrunk toward the layer's pooled
  M = Σ_r x xᵀ / R with κ = 256 rows: (H_e + 256 M) / (n_e + 256). down's rows are the expert's activations,
  recomputed from the quantised gate and up with Drama's rounding (gate and up to bf16, silu(gate) × up in fp32, one
  bf16 rounding). They are shrunk toward the expert's own mean diagonal times I, with the same κ. An expert's
  activation channels are its own neurons, so a pool across experts would mix unrelated coordinates. An expert no row
  reached gets I for down. Damping adds 1 % of the mean diagonal.
- **Encoding.** Columns go in natural order. At each 16-column group, `tf-drama-quant`'s scale search (the same
  rounding rule, h = the damped diagonal) runs on the weights as updated so far. Each column's rounding error is then
  fed forward through the upper Cholesky factor of H⁻¹. The global is the FP8 source's amax / (6 × 448) in f32; a
  nonzero projection whose global is not a positive finite f32 is refused. With a diagonal H, the bytes equal
  `tf-drama-quant`'s.
- **Precision.** The source, the weights and the search are f64, the search in Triton kernels without fused
  multiply-add. The Hessians, the Cholesky factors and the lazy updates of the columns past each 128-column batch are
  f32. Two runs on one machine write the same bytes.
- **Metadata.** One key, `gptq`, holds JSON with these fields:
  - `rows`: routed pairs per expert.
  - `kappa`, `damping` and `prior`: the gate/up and down shrinkage choices.
  - `output_error`: per weight, `{gptq, rtn}` = sqrt(tr(ΔHΔᵀ) / tr(WHWᵀ)) under the shrunk H. RTN is the same scale
    search without feedback. A 0 / 0 reports 0, and a nonzero over 0 reports `"undefined"`, as `tf-drama-quant` does.
- **Tests.** `python -m unittest drama_nvfp4_gptq_test` (in `tools/zig`) runs on the CPU in Triton's interpreter and
  hides every GPU, so it is safe beside a live server. `EQ_GPTQ_TEST_DEVICE=cuda` lets its CLI runs use the GPU.
- **Memory and time.** A layer's rows stay on the GPU as f32: 8 KiB a row, 1.4 GB at 174,529 rows. Next to them sit
  the 256 gate/up Hessians and their factors (4 GiB each) and f64 copies of the weights. Hessian and activation work
  grows linearly with the rows. On a GB10, one synthetic layer of 100,000 rows took 51 s at a 33.3 GiB peak.
