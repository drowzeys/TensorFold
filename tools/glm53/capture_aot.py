#!/usr/bin/env python3
"""Triton AOT capture for the Zig GLM-5.3 layers: oracle.py --record (no fixtures). Every kernel variant the first
N layers' decode and prompt windows launch (1- and 3-row decode windows, the 24-row prompt window, the radix select
at index bucket 32,768) is recorded, then packed into OUT/aot/{aot.json,cubins/,variants.txt}.
Usage (inside the image): TRITON_CACHE_DIR=<fresh dir> python -B capture_aot.py --model M --out OUT [--layers 4]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--record", *sys.argv[1:]]
    raise SystemExit(oracle.main())
