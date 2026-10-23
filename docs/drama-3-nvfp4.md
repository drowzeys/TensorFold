# Drama 3 NVFP4, pruned experts

How to build this branch and serve [`plotarmordev/drama-3-NVFP4-Pruned`](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned): Drama 3 with its routed experts stored in NVFP4 and only the kept experts in the checkpoint ([NVFP4 experts](drama-voices.md#nvfp4-experts), [compact experts](drama-voices.md#compact-experts)). Only this engine loads it; Transformers and SGLang-Omni do not.

## Prerequisites

- Linux with an NVIDIA GPU. The model has been built and run only on an NVIDIA GB10 (aarch64, compute capability 12.1). The kernels build for `sm_121` by default; `-Dsm=86,120,121` builds other targets, but the NVFP4 path is untested on them.
- Zig 0.17.0 exactly, from [ziglang.org/download](https://ziglang.org/download/) (the build refuses other versions).
- The CUDA toolkit's `nvcc` (this branch was built with CUDA 13.0) and a driver that runs it.
- The `hf` command from `huggingface_hub`, to download the model (about 22 GB).

## Build

```sh
git clone --branch drama-nvfp4 https://github.com/plotarmordev/TensorFold.git tensorfold
cd tensorfold
zig build native fatbins -Dnvcc=/usr/local/cuda/bin/nvcc -Dstrip=true -j4
```

This writes `zig-out/native/bin/tensorfold-native`, with the CUDA kernels embedded, and the kernel images in `zig-out/fatbin`. `-Dstrip=true` leaves out debug info, and with it the build machine's paths. The host-side tests need no GPU:

```sh
zig build test-cuda-host test-server-cpu -j4
```

## Serve

```sh
hf download plotarmordev/drama-3-NVFP4-Pruned --local-dir /path/to/drama-3-NVFP4-Pruned
zig-out/native/bin/tensorfold-native serve /path/to/drama-3-NVFP4-Pruned --name drama-3 --port 8080 --cfg-gamma 1.5 --cfg-batched
```

Serve this model with guidance γ 1.5 and batched guidance (`--cfg-gamma 1.5 --cfg-batched`, see [Guidance](drama-voices.md#guidance)): that is how it was measured. The startup log includes `routed experts NVFP4 in 40 of 40 layers`.

## Request

```sh
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "drama-3", "input": "The lighthouse keeper climbed the stairs as the storm rolled in.", "response_format": "wav"}' \
  -o speech.wav
```

The reply is a 44.1 kHz, 16-bit mono WAV file. With guidance on it carries `X-Cfg-Gamma`, the server's γ.

## Guidance needs a reference voice

`--cfg-gamma` guides only prompts that carry audio rows: a reference voice, or with `--long-form`, earlier segments. A request in the built-in voice, like the one above, has no reference to guide by, so it is drawn unguided, as at γ 1: the engine's synthesis log line says `cfg_gamma=1.00` and `unconditioned_runs=0`, although the reply's `X-Cfg-Gamma` header says 1.5. To have guidance act, give the request a reference voice in one of two ways:

- **A voice file.** Encode a clip of a voice you may use (your own, or one recorded with consent) into `NAME.json` as [Encode a clip](drama-voices.md#encode-a-clip) describes, put it in a directory, and serve with `--voices /path/to/voices --default-voice NAME`. A request then names it with `"voice": "NAME"`, or gets it by naming no voice.
- **Inline.** Send `"references": [{"text": "the clip's exact words", "vq_codes": [[...], ...]}]` in the request, where `vq_codes` is the `[10, frames]` code grid of a voice file. Audio itself is refused: send codes, not a clip.

```json
{"model": "drama-3", "input": "The lighthouse keeper climbed the stairs as the storm rolled in.", "voice": "NAME", "response_format": "wav"}
```

Voice files and clips hold a person's voice: keep them out of source control.
