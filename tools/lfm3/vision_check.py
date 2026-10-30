#!/usr/bin/env python3
"""vision_check.py REF.pt dump PIX.bin | compare OUT.bin: the CUDA tower's bf16 features against vision_ref64.py's float64 ones."""
import sys
import torch
ref = torch.load(sys.argv[1])
if sys.argv[2] == "dump":
    pv = ref["pixel_values"].to(torch.bfloat16).contiguous()
    open(sys.argv[3], "wb").write(pv.view(torch.int16).numpy().tobytes())
    t, gh, gw = ref["grid"][0].tolist()
    print(gh, gw)
else:
    f = ref["features"]
    import numpy as np
    raw = np.fromfile(sys.argv[3], dtype=np.int16)
    got = torch.from_numpy(raw).view(torch.bfloat16).to(torch.float64).view(f.shape)
    d = (got - f).abs()
    print(f"features {tuple(f.shape)}: max abs {d.max():.3e}, rel {(d.norm() / f.norm()):.3e}, cos {torch.nn.functional.cosine_similarity(got.flatten(), f.flatten(), 0):.6f}, finite {bool(torch.isfinite(got).all())}")
    rows = torch.nn.functional.cosine_similarity(got, f, 1)
    print(f"per-row cos: min {rows.min():.6f} mean {rows.mean():.6f}")
