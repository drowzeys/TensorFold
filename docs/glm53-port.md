# GLM-5.3 on four DGX Sparks: an unofficial CUDA port (this branch)

This branch is TensorFold 1.0.7 plus a Zig CUDA engine for GLM-5.3 (`glm_moe_dsa`, EXL3 weights) at tensor parallel 4:
one rank a DGX Spark (GB10), the ranks joined over RoCE. It lives on a fork. It is not part of upstream TensorFold and
upstream has not reviewed or tested it; TensorFold itself is by Ash Hart and its contributors, and everything outside
the files listed below is theirs, unchanged.

## What the branch adds

| Part | Where |
|---|---|
| The family: load, forward, drafting (MTP, DSpark, DFlash2), sampling, prompt reuse, concurrent streams, decode context parallelism | `zig/src/families/glm_moe_dsa/` |
| Its kernels | `zig/kernels/cuda/glm53_*.cu` (the EXL3 kernels are device-only copies of exllamav3's, by turboderp) |
| `tensorfold-native serve` for the family (rank 0 serves, ranks 1..3 follow) and the same engine behind a C ABI (`libtensorfold_glm53.so`) | `zig/src/native/glm53_cuda.zig`, `glm53_capi.zig` |
| The gate tool, with the as-served scorer | `zig/tests/cuda/glm53_generate.zig` (`tf-glm53-generate`), `docs/glm53-score.md` |
| Off-by-default settings: a 6-bit output head, a local health check, the exchange trace, bf16 exchanges, an EXL3 `kv_b` decoded at load | `docs/glm53-speed.md` |
| Gate and reference scripts as they were run on the author's cluster | `tools/glm53/` |

## Which published image is which commit

| Image | Built from |
|---|---|
| `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-mlx-serve` | this branch's tree, with one difference stated below |

The image's engine was built from a commit of the working branch (`a4d016d4`, recorded in the image's
`/opt/keys-glm53/VERSION`) that also held an experimental routed-expert pruning setting. That setting was never switched
on in the image, was ruled out, and is not in this branch: `prune.zig`, its test fixture, its reference script, the
kernel `glm53_expert_prune` and the lines that read `TF_GLM53_EXPERT_*` are removed. Nothing else in the source
differs apart from documentation and the default paths in `tools/glm53/`. Built side by side, 37 of the 40 kernel packs
are byte for byte the image build's; `glm53_decode` differs by that one kernel, and `glm53_exl3_experts` and
`glm53_prompt_experts` (unchanged sources, compiled with `-lineinfo`) differ in their line information, which holds the
checkout's path. The image's HTTP server on rank 0 is mlx-serve (by David Dalcu) with a bridge to
`libtensorfold_glm53.so`; that bridge is not in this repository.

## Building

Zig 0.17.0 and a CUDA 13 toolkit, for `sm_121` (GB10):

```sh
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe fatbins            # the kernel packs
zig build -Dfatbins=zig-out/fatbin -Dsm=121 -Doptimize=safe native glm53-lib tf-glm53-generate
zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe test-glm53 test-cuda-host   # host tests, no GPU
```

The engine also needs the family's precompiled Triton kernel pack and tile tables for the context and stream count it
serves; the published image carries them for 32,768 tokens x 4 streams, 65,536 x 2 and 1,000,000 x 1.

## Measured

From the published image on four DGX Sparks, the 2.75 bpw EXL3 pack, the DSpark drafter, thinking on (the recipe
repository keeps the logs):

| Shape | Prose tok/s | Code tok/s |
|---|---|---|
| 32,768 x 4, one stream | 39.6 | 57.8 |
| 32,768 x 4, four streams together | 68.5 | 80.5 |
| 1,000,000 x 1 | 36.0 | 48.4 |

Each shape was loaded once. As served, the pack scores KL 0.1756 against the unquantized checkpoint (0.1757 with the
6-bit head); `docs/glm53-score.md` has the sets.

## Limits

- GLM-5.3 at world 4 on GB10 only; no other machine or rank count has been run.
- Several streams need decode context parallelism off, so each stream holds its whole cache on every rank: 32,768 x 4
  and 65,536 x 2 are the measured shapes.
- A context or stream count outside the shapes above needs kernels the pack does not hold.
- `kv_b` is read as bf16 in a window; `TF_GLM53_KVB_Q` only shrinks a pack that holds it as an EXL3 linear.
- bf16 exchanges (`TF_GLM53_ROCE_BF16`) are faster and are left off: they moved the tool-call rows' KL by 0.0026.
