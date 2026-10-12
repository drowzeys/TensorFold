#!/usr/bin/env python3
"""Fixtures for the Zig GLM-5.3 layers: oracle.py --fixtures (no recording). Layers 0..N-1 at world 1 through the
fused path's layer-slice loader (weights.RankReader / load_layer); per window and layer the output rows, cache rows,
selections and end-of-layer scratch -> OUT/fixtures/{meta.json,common,short,long}.safetensors.
Usage (inside the image): python -B dump_fixtures.py --model M --out OUT [--layers 4] [--long 20000]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--fixtures", *sys.argv[1:]]
    raise SystemExit(oracle.main())
