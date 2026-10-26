# libbolt for CUDA (Nemotron 3.5 Bolt 74B-A9B with NVIDIA's LiLiCorr drafter)

`bolt_*.cu` are MiaAI-Lab's libbolt from [Nemotron-3.5-Bolt-TensorFold-DGX-Spark](https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark)
at 89e1c8a (MIT, `LICENSE-MiaAI-Lab`; speed work by ChriZzn). `zig/src/families/nemotron/bolt_native.zig` and
`bolt_api.zig` are its lane backend, here on TensorFold 1.1.5 with Mia's registration patch rebased. Weights:
nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB and nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr (NVIDIA).

Changes over the recipe, in bolt_native.zig only (no arithmetic changes):
- the first token leaves before LiLiCorr's first proposal (`first_before_draft`; `BOLT_FIRST_EARLY=0` turns it off);
- a prompt-cache mark within `BOLT_TAIL_MERGE` (default 8) rows of the prompt's end is kept at the end, so a chat
  turn's tail is one forward instead of two; `BOLT_MARK_LATE=1` copies that state after the first token instead.

Build: `zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dbolt-src=$PWD/zig/bolt-cuda -Doptimize=fast install native bolt`.

Serve: `TENSORFOLD_BOLT_LIB=<prefix>/native/lib/libbolt.so BOLT_DRAFT=<LiLiCorr dir> BOLT_SSM_LAZY=1 tensorfold-native serve <Bolt dir> ...`
