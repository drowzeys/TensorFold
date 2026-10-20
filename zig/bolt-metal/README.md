# Nemotron 3.5 Bolt on Metal

NVIDIA's [Nemotron 3.5 Bolt 74B-A9B, 24 GB packed checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
(revision `a326822a`) and its [LiLiCorr drafter](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)
(revision `4c6a5613`) on Apple GPUs, served by `tensorfold-native` with TensorFold driving the speculation.

This is a Metal port of MiaAI-Lab's DGX Spark recipe,
[Nemotron-3.5-Bolt-TensorFold-DGX-Spark](https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark). Its
design is kept as it is, and so is its C ABI, so its lane backend runs on Metal nearly line for line:

- **Weights stay packed**: IQ2_XXS experts, NVFP4 Mamba projections, Q8_0 attention, latent projections and head,
  about 5 GB read per token at batch one. They are repacked at load into planar layouts (int8 rows and half scales for
  Q8_0, 64-byte groups and half scales for IQ2_XXS) and decoded in registers. Activations stay fp32.
- **Cheap rollback**: two SSM state copies per sequence; a verify window writes only its final state, and `keep(n)`
  flips a pointer or replays `n` rows from the saved scan inputs. The conv state is a ring of nine slots.
- **TensorFold drives the drafts** through the backend's `verify`, `keep` and `draft`: LiLiCorr proposes seven tokens a
  round, and TensorFold's cost rule picks the verify width from window and drafter costs measured at load.
- **Prompt cache**: Mamba, attention and drafter state are saved at the cache's marks and restored for a shared prefix.
- **Batch-invariant decode**: every row of a decode or verify block sums in the same order whatever the block holds,
  and the attention key split follows the sequence's own length, so drafted replies equal plain ones.
- The three drafter bugs Mia's engine fixed are fixed here too: the shared expert reads the block input's norm, Mamba
  `dt` is not clamped, and the drafter's grouped convolution follows `o_proj`.

## Layout

| Path | What |
| --- | --- |
| `zig/bolt-metal/bolt_metal.mm` | libbolt on Metal: loading, the block forward, the drafter, sequences, snapshots, the C ABI |
| `zig/bolt-metal/bolt.metal` | the kernels: packed GEMVs and prompt GEMMs, routed experts, Mamba conv/scan/replay, attention, router, top-k, the drafter's |
| `zig/bolt-metal/bolt_args.h` | argument blocks shared by the host and the kernels |
| `zig/bolt-metal/bolt_test.mm` | parity against fp32 reference logits, generation and prompt benchmarks |
| `zig/bolt-metal/api_check.py` | OpenAI API checks (chat, streaming, forced tool calls, JSON schema) and decode timing |
| `zig/src/families/nemotron/bolt_metal.zig` | the lane backend (Mia's `bolt_native.zig` without the CUDA host's pieces) |
| `zig/src/native/bolt_host.zig` | the Metal host: lane core, scheduler and prompt cache around the backend |

The kernels are compiled from source embedded in the library when it loads, so building needs no Metal toolchain;
`tensorfold-native` itself does (Xcode's).

## Build and serve

```bash
zig/bolt-metal/build.sh                                  # build/libbolt.dylib and build/bolt_test
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer zig build native -Dcpu=apple_m1 -Doptimize=ReleaseFast
zig/bolt-metal/serve.sh                                  # :8093, model id nvidia-nemotron-3.5-bolt
```

`serve.sh` reads `MODEL` (the checkpoint folder; the drafter is found as `LiLiCorr` or
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr` beside it, or `BOLT_DRAFT`), `PORT`, `CONTEXT`, `PARALLEL` and
`DRAFTS=0`. Thinking is off by default; a request turns it on with `chat_template_kwargs`. Structured output and
forced tool calls need `libtfgrammar.dylib` beside the server (`zig build grammar -Dxgrammar=DIR`).

## Checks

Measured on an M5 Ultra (256 GB), single process, greedy.

Parity against fp32 reference logits (teacher-forced, NVIDIA's checkpoint run in fp32 by `tools/zig/bolt_reference.py`):

| Prompt | Path | Positions | KL mean | KL max | Top-1 |
| --- | --- | ---: | ---: | ---: | ---: |
| code | one-token steps | 304 | 0.00016 | 0.016 | 304/304 |
| code | prefill, then 8-row verify windows kept whole or cut to 5 | 255 | 0.00008 | 0.002 | 255/255 |
| shots | one-token steps | 197 | 0.00026 | 0.016 | 197/197 |
| shots | prefill, then verify windows | 125 | 0.00013 | 0.002 | 125/125 |

Decode through `tensorfold-native serve` (`api_check.py`, 512 tokens, thinking off; prompts: a cyberpunk heist story, a
street-dance battle bracket in Python, a sci-fi chase shot list), against llama.cpp Metal on the same Mac:

| | Prose | Code | Shots |
| --- | ---: | ---: | ---: |
| TensorFold Metal (LiLiCorr drafts, cost rule) | 72.2 tok/s | 100.5 tok/s | 89.5 tok/s |
| llama.cpp Metal (Q8 Mamba GGUF) | 56.8 tok/s | 57.1 tok/s | 57.2 tok/s |

A plain step (no drafts) takes 12.6 ms (79 tok/s); verify windows of 1, 2, 4 and 8 rows take 12.9, 16.8, 22.0 and
40.2 ms, a proposal of seven drafts 7.9 ms. With a fixed width of eight, LiLiCorr lands 4.3 drafts a round on code,
2.1 on the shot list and 1.5 on prose. Two streams together reach 87 tok/s in total, four 93.

Prompt pass: 922 tok/s at 1,024 tokens, 862 at 8,192, 678 at 32,768.

## Credits

- MiaAI-Lab's [Nemotron-3.5-Bolt-TensorFold-DGX-Spark](https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark):
  the engine design, the C ABI, the lane backend and the drafter bug fixes this port follows.
- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and its contributors: the server, lane core and scheduler.
- NVIDIA: the checkpoint and the LiLiCorr drafter (OpenMDW License Agreement 1.1; not part of this repository).
- [ggml](https://github.com/ggml-org/ggml) (MIT): the IQ2_XXS and Q8_0 formats and the IQ2_XXS grid table.
