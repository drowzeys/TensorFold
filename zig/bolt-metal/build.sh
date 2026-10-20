#!/bin/bash
# Build libbolt.dylib (Nemotron 3.5 Bolt on Metal) and its test tool. The kernels are compiled at load from the
# source embedded in the library, so no Metal toolchain is needed.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-build}"
mkdir -p "$OUT"
{
    printf 'R"BOLTSRC('
    cat bolt_args.h iq2_grid.h bolt.metal
    printf ')BOLTSRC"\n'
} > "$OUT/bolt_src.inc"
clang++ -std=c++17 -O2 -fobjc-arc -fPIC -shared -I"$OUT" -I. -o "$OUT/libbolt.dylib" bolt_metal.mm \
    -framework Metal -framework Foundation -install_name @rpath/libbolt.dylib
clang++ -std=c++17 -O2 -o "$OUT/bolt_test" bolt_test.mm -L"$OUT" -lbolt -Wl,-rpath,@loader_path
echo "built $OUT/libbolt.dylib $OUT/bolt_test"
