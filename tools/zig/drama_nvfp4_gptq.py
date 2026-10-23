#!/usr/bin/env python3
"""Drama 3 routed experts to NVFP4 by GPTQ on the probe's --rows dumps: per layer, its C1 tensors in one file."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

# without a GPU the kernels run in Triton's interpreter, which triton picks up as it loads
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

EXPERTS, TOP_K, LAYERS = 256, 8, 40
GROUP, FP8_BLOCK = 16, 128
BATCH = 128  # columns between GPTQ's lazy updates of the rest of W
KAPPA, DAMPING = 256, 0.01
PREFIX = "model.language_model.layers."
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# the 8 e2m1 magnitudes, then the 7 midpoints between neighbours
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float64)
SCALES = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).double()


@triton.jit
def _nearest(a, s, g, e2m1_ptr):
    """Slice B's C2 rounding of magnitudes a: index past each midpoint (m s) g, on it when the upper index is even."""
    idx = (a > (tl.load(e2m1_ptr + 8) * s) * g).to(tl.int32)
    for j in tl.static_range(1, 7):
        t = (tl.load(e2m1_ptr + 8 + j) * s) * g
        if j % 2 == 1:
            idx += (a >= t).to(tl.int32)
        else:
            idx += (a > t).to(tl.int32)
    return idx, a - (tl.load(e2m1_ptr + idx) * s) * g


@triton.jit
def _search(a_ptr, h_ptr, g_ptr, scale_ptr, e2m1_ptr, code_ptr, rows, BLOCK: tl.constexpr):
    """C2 step 4 per row of 16: of e4m3 codes 1..126 the least Σ h (|W| - rec)² in column order, smaller on ties."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = r < rows
    c = tl.arange(0, 128)
    valid = (c >= 1) & (c <= 126)
    s = tl.load(scale_ptr + c, mask=valid, other=1.0)[None, :]
    g = tl.load(g_ptr + r, mask=ok, other=1.0)[:, None]
    loss = tl.zeros([BLOCK, 128], tl.float64)
    for i in tl.static_range(16):
        _, d = _nearest(tl.load(a_ptr + r * 16 + i, mask=ok, other=0.0)[:, None], s, g, e2m1_ptr)
        loss = loss + (tl.load(h_ptr + r * 16 + i, mask=ok, other=0.0)[:, None] * d) * d
    loss = tl.where(valid[None, :], loss, float("inf"))
    tl.store(code_ptr + r, tl.argmin(loss, 1, tie_break_left=True), mask=ok)


@triton.jit
def _round(a_ptr, s_ptr, g_ptr, e2m1_ptr, idx_ptr, rows, BLOCK: tl.constexpr):
    """C2 rounding of one column's magnitudes at their row's chosen scale."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = r < rows
    a = tl.load(a_ptr + r, mask=ok, other=0.0)
    idx, _ = _nearest(a, tl.load(s_ptr + r, mask=ok, other=1.0), tl.load(g_ptr + r, mask=ok, other=1.0), e2m1_ptr)
    tl.store(idx_ptr + r, idx.to(tl.uint8), mask=ok)


def _launch(kernel, rows, block, *args):
    # the interpreter runs each program in numpy, so on the CPU one program takes every row
    block = block if args[0].is_cuda else triton.next_power_of_2(rows)
    kernel[(triton.cdiv(rows, block),)](*args, rows, BLOCK=block, enable_fp_fusion=False)


def nearest(a, s, g):
    """e2m1 magnitude indices (uint8) of magnitudes a at scale values s and globals g, all f64 of one shape."""
    idx = torch.empty(a.shape, dtype=torch.uint8, device=a.device)
    _launch(_round, a.numel(), 256, a.contiguous(), s.contiguous(), g.contiguous(), E2M1.to(a.device), idx)
    return idx


def global_scale(w, name):
    """C2 step 3 per expert of W [E][n][k]: g = f32(amax|W| / (6 × 448)) as f64, 1 for an all-zero W."""
    amax = w.abs().amax((1, 2))
    g = torch.where(amax > 0, (amax / (6 * 448)).float().double(), 1.0)
    bad = ~(g.isfinite() & (g > 0))
    if bad.any():
        e = int(bad.nonzero()[0])
        a, ge = amax[e].item(), g[e].item()
        raise ValueError(f"{name} expert {e}: amax {a!r} gives the f32 global {ge!r}, not a positive finite one")
    return g


def shrink(hess, count, prior):
    """H_e' = (H_e + κ P_e) / (n_e + κ): κ rows of the prior keep a rarely routed expert's Hessian well posed."""
    return (hess + KAPPA * prior) / (count.to(hess) + KAPPA)[:, None, None]


def damp(hess):
    """H with 1 % of its mean diagonal added to the diagonal; a Hessian without energy becomes I (C3's h ≡ 1)."""
    out = hess.clone()
    mean = hess.diagonal(dim1=1, dim2=2).mean(1)
    out.diagonal(dim1=1, dim2=2).add_((DAMPING * mean)[:, None])
    out[mean == 0] = torch.eye(hess.shape[1], dtype=hess.dtype, device=hess.device)
    return out


def upper(damped):
    """GPTQ's feedback matrix: the upper Cholesky factor of H⁻¹."""
    return torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(damped)), upper=True)


def quantise(w, h, g, feedback=None):
    """W [E][n][k] by columns into C1 nibbles, scale codes and Q; GPTQ when `feedback` is the upper factor of H⁻¹."""
    w, g = w.clone(), g.contiguous()
    experts, n, k = w.shape
    device = w.device
    nib = torch.empty(experts, n, k, dtype=torch.uint8, device=device)
    codes = torch.empty(experts, n, k // GROUP, dtype=torch.uint8, device=device)
    q = torch.empty_like(w)
    e2m1, scales = E2M1.to(device), SCALES.to(device)
    for b0 in range(0, k, BATCH):
        b1 = min(b0 + BATCH, k)
        err = torch.empty(experts, n, b1 - b0, dtype=w.dtype, device=device)
        for c in range(b0, b1):
            if c % GROUP == 0:
                # the group's scales come from the C2 search on the weights as the feedback has updated them so far
                a = w[:, :, c:c + GROUP].abs()
                code = torch.empty(experts * n, dtype=torch.int32, device=device)
                hh = h[:, None, c:c + GROUP].expand(experts, n, GROUP).contiguous()
                _launch(_search, experts * n, 4, a.contiguous(), hh, g, scales, e2m1, code)
                code = code.view(experts, n).masked_fill(a.amax(-1) == 0, 0)
                codes[:, :, c // GROUP] = code.to(torch.uint8)
                s = scales[code]
            x = w[:, :, c]
            idx = nearest(x.abs(), s, g).masked_fill(code == 0, 0)
            mag = (e2m1[idx.long()] * s) * g
            qc = torch.where(x < 0, -mag, mag)
            q[:, :, c] = qc
            nib[:, :, c] = idx | ((x < 0) & (idx > 0)).to(torch.uint8) << 3
            if feedback is not None:
                e = (x - qc) / feedback[:, c, c, None]
                w[:, :, c + 1:b1] -= e[..., None] * feedback[:, None, c, c + 1:b1]
                err[:, :, c - b0] = e
        if feedback is not None:
            w[:, :, b1:] -= (err.float() @ feedback[:, b0:b1, b1:]).double()
    return nib[..., 0::2] | nib[..., 1::2] << 4, codes, q


def output_error(w, q, hess):
    """Per expert sqrt(tr(ΔHΔᵀ) / tr(WHWᵀ)), Δ = W - Q: 0 for 0 / 0, "undefined" (as tf-drama-quant) for nonzero / 0."""
    d, wf = (w - q).float(), w.float()
    num = ((d @ hess) * d).sum((1, 2), dtype=torch.float64).tolist()
    den = ((wf @ hess) * wf).sum((1, 2), dtype=torch.float64).tolist()
    if not all(math.isfinite(v) and v >= 0 for v in num + den):
        raise ValueError("output error: a non-finite or negative tr(ΔHΔᵀ) or tr(WHWᵀ)")
    return [math.sqrt(a / b) if b else (0.0 if a == 0 else "undefined") for a, b in zip(num, den)]


def finite(t, what):
    """t, refused unless every value is finite."""
    if not t.isfinite().all():
        raise ValueError(f"{what}: a non-finite value")
    return t


def gram(x):
    """Σ x xᵀ over the rows of x."""
    return x.T @ x


def gate_up_hessians(x, picks):
    """Per expert Σ x xᵀ over its rows, pair count and rows, and the pooled Σ_r x xᵀ / R; no routing weight enters."""
    flat = picks.reshape(-1).cpu()
    count = torch.bincount(flat, minlength=EXPERTS)
    members = [m.to(x.device) for m in torch.split(torch.argsort(flat, stable=True) // TOP_K, count.tolist())]
    hess = finite(torch.stack([gram(x[m]) for m in members]), "gate/up Hessian")
    return hess, count.to(x.device), members, finite(gram(x) / len(x), "pooled gate/up Hessian")


def down_hessians(x, members, qg, qu):
    """Per expert Σ a aᵀ over its rows of the act a from the quantised gate Qg and up Qu [E][n][k]."""
    out = []
    for e, m in enumerate(members):
        # Drama's SwiGLU: gate and up rounded to bf16, silu(gate)·up in fp32, one bf16 rounding
        xe = x[m]
        gp, upp = (xe @ qg[e].T).bfloat16().float(), (xe @ qu[e].T).bfloat16().float()
        act = (gp / (1 + torch.exp(-gp)) * upp).bfloat16().float()
        for t in (gp, upp, act):
            finite(t, f"expert {e}: down activation")
        out.append(gram(act))
    return finite(torch.stack(out), "down Hessian")


def encode(w, hess, g):
    """GPTQ of W under the shrunk H `hess`, and the RTN Q; both search scales with h = the damped diagonal."""
    damped = damp(hess)
    h = damped.diagonal(dim1=1, dim2=2).double()
    feedback = upper(damped)
    del damped
    gptq = quantise(w, h, g, feedback)
    del feedback
    return gptq, quantise(w, h, g)[2]


def decode(name, codes, scale):
    """C2 step 1: f64 W = e4m3 code × weight_scale_inv of its 128 × 128 block; refuses non-finite W, scales ≤ 0."""
    n, k = codes.shape
    blocks = (-(-n // FP8_BLOCK), -(-k // FP8_BLOCK))
    if codes.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or tuple(scale.shape) != blocks:
        raise ValueError(f"{name}weight: F8_E4M3 with an F32 weight_scale_inv {list(blocks)}")
    s = scale.double()
    if not (s.isfinite() & (s > 0)).all():
        raise ValueError(f"{name}weight_scale_inv: a non-finite or non-positive scale")
    w = codes.double() * s.repeat_interleave(FP8_BLOCK, 0)[:n].repeat_interleave(FP8_BLOCK, 1)[:, :k]
    if not w.isfinite().all():
        raise ValueError(f"{name}weight: a non-finite value")
    return w


def source(src, layer, device):
    """The layer's FP8 routed experts in SRC as f64 W [256][n][k] per projection (gate, up, down)."""
    shards = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def tensor(name):
        if name not in shards:
            raise ValueError(f"{src}: no {name}")
        if shards[name] not in files:
            files[shards[name]] = safe_open(src / shards[name], "pt")
        return files[shards[name]].get_tensor(name).to(device)

    out = []
    for proj in PROJECTIONS:
        names = [f"{PREFIX}{layer}.mlp.experts.{e}.{proj}." for e in range(EXPERTS)]
        out.append(torch.stack([decode(m, tensor(m + "weight"), tensor(m + "weight_scale_inv")) for m in names]))
    return out


def routed(rows, layer, width, config):
    """C3 --rows of a layer, checked against rows.json and SRC's config.json sha256: x [R][width] bf16, picks [R][8]."""
    manifest = json.loads((rows / "rows.json").read_text())
    count = manifest.get("rows")
    if manifest.get("schema") != "drama-probe-rows/1" or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{rows}/rows.json: drama-probe-rows/1 with a positive row count")
    if manifest.get("identity", {}).get("config_sha256") != config:
        raise ValueError(f"{rows}/rows.json: rows of a checkpoint whose config.json sha256 is not {config}")
    if not isinstance(manifest.get("layers"), int) or layer >= manifest["layers"]:
        raise ValueError(f"{rows}/rows.json: no layer {layer}")
    files = {"class": ("class.u8", "u8", [count], 1), "x": ("layer-LL.x.bf16", "bf16", [count, width], 2),
             "picks": ("layer-LL.picks.i32", "i32", [count, TOP_K], 4)}
    paths = {}
    for key, (file, dtype, shape, size) in files.items():
        entry = manifest.get(key)
        if not isinstance(entry, dict) or [entry.get(f) for f in ("file", "dtype", "shape")] != [file, dtype, shape]:
            raise ValueError(f"{rows}/rows.json: {key} is not {file} {dtype} {shape}")
        paths[key] = rows / file.replace("LL", f"{layer:02d}")
        if paths[key].stat().st_size != math.prod(shape) * size:
            raise ValueError(f"{paths[key]}: not {dtype} {shape}")
    if (np.fromfile(paths["class"], dtype=np.uint8) > 2).any():
        raise ValueError(f"{paths['class']}: a class outside 0 prompt_text, 1 prompt_audio, 2 generated")
    p = torch.from_numpy(np.fromfile(paths["picks"], dtype="<i4").reshape(count, TOP_K)).long()
    if p.min() < 0 or p.max() >= EXPERTS:
        raise ValueError(f"{paths['picks']}: an expert outside 0 to {EXPERTS - 1}")
    ordered = p.sort(1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError(f"{paths['picks']}: a row picks one expert twice")
    x = torch.from_numpy(np.fromfile(paths["x"], dtype="<i2")).view(torch.bfloat16).reshape(count, width)
    return finite(x, paths["x"]), p


def mean_of(errors):
    """The mean of the defined errors (NaN when none is)."""
    defined = [v for v in errors if v != "undefined"]
    return sum(defined) / len(defined) if defined else math.nan


def encode_layer(src, rows, layer, device):
    """One layer's C1 tensors, the file metadata, and a summary line."""
    gate, up, down = source(src, layer, device)
    _, inner, hidden = gate.shape
    if up.shape != gate.shape or down.shape != (EXPERTS, hidden, inner) or hidden % GROUP or inner % GROUP:
        raise ValueError(f"{src}: layer {layer}: gate/up [n][k] and down [k][n] with n and k multiples of {GROUP}")
    scale2 = [global_scale(w, p) for w, p in zip((gate, up, down), PROJECTIONS)]
    x, picks = routed(rows, layer, hidden, hashlib.sha256((src / "config.json").read_bytes()).hexdigest())
    x = x.to(device).float()
    # gate/up: all routed pairs pooled, H_pool / N_pool = Σ_r x xᵀ / R since every row is routed to 8 experts
    hess, count, members, pool = gate_up_hessians(x, picks)
    hess = shrink(hess, count, pool)
    g = torch.cat([scale2[0][:, None].expand(-1, inner), scale2[1][:, None].expand(-1, inner)], 1)
    (gu_nib, gu_codes, q), rtn = encode(torch.cat([gate, up], 1), hess, g)
    errors = [(output_error(gate, q[:, :inner], hess), output_error(gate, rtn[:, :inner], hess)),
              (output_error(up, q[:, inner:], hess), output_error(up, rtn[:, inner:], hess))]
    del rtn
    hess = down_hessians(x, members, q[:, :inner].float(), q[:, inner:].float())
    del q
    # down: C3's uniform prior, the expert's mean diagonal (an expert's act channels are its own)
    mean = torch.where(count > 0, hess.diagonal(dim1=1, dim2=2).sum(1) / (inner * count.clamp(min=1)), 1.0)
    hess = shrink(hess, count, mean[:, None, None] * torch.eye(inner, device=device))
    (dn_nib, dn_codes, q), rtn = encode(down, hess, scale2[2][:, None].expand(-1, hidden))
    errors.append((output_error(down, q, hess), output_error(down, rtn, hess)))
    parts = [(gu_nib[:, :inner], gu_codes[:, :inner]), (gu_nib[:, inner:], gu_codes[:, inner:]), (dn_nib, dn_codes)]
    tensors, table = {}, {}
    for proj, (nib, codes), g, (err, base) in zip(PROJECTIONS, parts, scale2, errors):
        nib, codes, g = nib.cpu(), codes.cpu(), g.float().cpu()
        for e in range(EXPERTS):
            name = f"{PREFIX}{layer}.mlp.experts.{e}.{proj}."
            tensors[name + "weight"] = nib[e].clone()
            tensors[name + "weight_scale"] = codes[e].clone().view(torch.float8_e4m3fn)
            tensors[name + "weight_scale_2"] = g[e].clone()
            table[name + "weight"] = {"gptq": err[e], "rtn": base[e]}
    prior = {"gate_up": "pooled: sum of x x^T over all rows / R", "down": "uniform: the expert's mean diagonal times I"}
    # one key: safetensors writes metadata keys in hash order, which differs from run to run
    metadata = {"gptq": json.dumps({"rows": count.tolist(), "kappa": KAPPA, "damping": DAMPING, "prior": prior,
                                    "output_error": table}, allow_nan=False)}
    means = ", ".join(f"{p} {mean_of(e):.4f}/{mean_of(r):.4f}" for p, (e, r) in zip(PROJECTIONS, errors))
    return tensors, metadata, f"{len(x)} rows, mean output error GPTQ/RTN {means}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="the FP8 Drama 3 checkpoint (read only)")
    parser.add_argument("rows", type=Path, help="tf-drama-probe --rows DIR")
    parser.add_argument("out", type=Path, help="gets layer-LL.safetensors for every layer")
    parser.add_argument("--layers", default=",".join(map(str, range(LAYERS))), help="comma separated, default all")
    args = parser.parse_args()
    layers = args.layers.split(",")
    if not all(layer.isdigit() and int(layer) < LAYERS for layer in layers):
        parser.error(f"--layers: layer indices from 0 to {LAYERS - 1}, comma separated")
    layers = [int(layer) for layer in layers]
    # cuBLAS repeats its sums bit for bit only with a fixed workspace
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        start = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            tensors, metadata, summary = encode_layer(args.src, args.rows, layer, device)
        except (OSError, ValueError, KeyError) as e:
            sys.exit(f"drama_nvfp4_gptq.py: layer {layer}: {e}")
        path = args.out / f"layer-{layer:02d}.safetensors"
        save_file(tensors, path, metadata)
        peak = f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if device == "cuda" else ""
        print(f"{path}: {summary}, {time.time() - start:.1f} s{peak}", flush=True)


if __name__ == "__main__":
    main()
