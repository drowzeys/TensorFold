#!/bin/bash
# Drama 3 port build on .4 (host toolchain): drama3.fatbin with nvcc (torch's flags + -O3, sm_121), the other fatbins
# prebuilt (the glm53-prefill image build), then drama3-generate / drama3-lib / test-drama3 with the local zig.
set -e
T=~/zig-port/tf-drama3; B=~/zig-port/d3build
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121 \
  -o $B/fatbin/drama3.fatbin $T/zig/kernels/cuda/drama3.cu
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121 \
  -diag-suppress 177 -o $B/fatbin/drama3_fp8.fatbin $T/zig/kernels/cuda/drama3_fp8.cu
nvcc -fatbin -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
  -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -std=c++20 -O3 -gencode=arch=compute_121,code=sm_121 \
  -o $B/fatbin/drama3_codec.fatbin $T/zig/kernels/cuda/drama3_codec.cu
cd $T
export ZIG_LOCAL_CACHE_DIR=$B/cache/zig-local ZIG_GLOBAL_CACHE_DIR=$B/cache/zig-global
nice -n 10 ~/opt/zig/zig build -Dfatbins=$B/fatbin -Dsm=121 -Doptimize=${OPT:-ReleaseFast} --prefix $B/zig-out -j8 ${@:-drama3-generate}
