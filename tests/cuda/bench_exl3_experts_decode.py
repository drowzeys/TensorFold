"""Decode-window timing of the routed EXL3 experts (``cuda/exl3/experts``) on full GLM-5.3's per-rank (TP=4) shapes:
256 experts, 6144 -> 512 (gate|up as column blocks of one fused trellis, experts_cx's layout) -> 6144, top-8, mcg
codebook, a width per expert of 2 / 3 / 4 bits (2.75 bpw on average like the served checkpoint's mix is close
enough for bytes), bf16 SwiGLU. Per knob setting and window of R rows: microseconds a MoE layer's routed chain
(group .. combine, with the shared-expert add folded or not) and the trellis GB/s it reads.

Each timed graph holds CALLS routed calls with different picks (so the experts' words come from DRAM, as in a
forward, not from L2); repeats 3 times, prints the median per call. Every setting's output is also checked
bit-equal to the first one's.

    python tests/cuda/bench_exl3_experts_decode.py [--rows 1 3 8 16 32] [--settings 0,0,0 1,1,1 2,1,1 1,3,1]

A setting is loads,fuse,pdl (TF_EXL3_EXPERTS_LOADS / _FUSE / _PDL); "0,0,0" is the chain before these patches.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from tensorfold.cuda.exl3 import experts

E, D, I, TOPK = 256, 6144, 512, 8
CALLS = 24


def layer(seed: int = 1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    it = I // 16
    gate, up, down = [], [], []
    for e in range(E):
        k2 = (4, 6, 8, 6)[e % 4]                         # 2, 3, 4, 3 bits
        fused = torch.randint(-2**15, 2**15, (D // 16, 2 * it, 8 * k2), dtype=torch.int16, device="cuda",
                              generator=g)
        dt = torch.randint(-2**15, 2**15, (I // 16, D // 16, 8 * k2), dtype=torch.int16, device="cuda", generator=g)

        def sc(n, m):
            return ((torch.rand((n,), device="cuda", generator=g) + 0.5) * m).half()

        gate.append((fused[:, :it], sc(D, 0.013), sc(I, 1.0)))
        up.append((fused[:, it:], sc(D, 0.013), sc(I, 1.0)))
        down.append((dt, sc(I, 0.044), sc(D, 0.25)))
    return experts.prepare(gate, up, down, "mcg", gu_stride=2 * it)


def picks(R: int, n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        sel = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(R)]).to(torch.int32).cuda()
        w = (torch.rand((R, TOPK), generator=g) * 0.2 + 0.05).float().cuda()
        out.append((sel.contiguous(), w.contiguous()))
    return out


def time_setting(ex, s, x, ps, sy, out, R, loads, fuse, pdl, fold) -> tuple[float, torch.Tensor]:
    kw = dict(loads=loads, fuse=fuse, pdl=pdl)

    def body():
        for sel, w in ps:
            if fold:
                experts.routed(x, sel, w, ex, s, out, R, act_mode=experts.ACT_BF16, sy=sy, **kw)
            else:
                experts.routed(x, sel, w, ex, s, out, R, act_mode=experts.ACT_BF16, **kw)
                out.add_(sy)

    body()
    torch.cuda.synchronize()
    first = out.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(3):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(5):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / (5 * len(ps)))
    del g
    return statistics.median(ts), first


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 3, 8, 16, 32])
    ap.add_argument("--settings", nargs="+", default=["0,0,0", "1,0,0", "1,1,0", "1,1,1", "2,1,1", "4,1,1", "1,3,1"])
    ap.add_argument("--no-fold", action="store_true", help="time the separate shared-expert add (out += sy)")
    args = ap.parse_args()
    ex = layer()
    print(f"routed experts: {E} x (6144 -> 512 -> 6144) mcg 2/3/4 bits, top-{TOPK}, {CALLS} calls a graph")
    for R in args.rows:
        s = experts.Scratch(ex, R, TOPK)
        g = torch.Generator(device="cuda").manual_seed(R)
        x = (torch.randn((R, D), device="cuda", generator=g) * 0.5).to(torch.bfloat16)
        sy = torch.randn((R, D), device="cuda", generator=g).float()
        out = torch.empty((R, D), dtype=torch.float32, device="cuda")
        ps = picks(R, CALLS, seed=R)
        nbytes = sum(ex.nbytes_read(sorted(set(sel.flatten().tolist()))) for sel, _ in ps) / len(ps)
        ref = None
        for st in args.settings:
            loads, fuse, pdl = (int(v) for v in st.split(","))
            us, first = time_setting(ex, s, x, ps, sy, out, R, loads, fuse, pdl, not args.no_fold)
            same = "" if ref is None else ("  bits ok" if torch.equal(first.view(torch.int32), ref.view(torch.int32))
                                           else "  BITS DIFFER")
            if ref is None:
                ref = first
            print(f"R={R:2d} loads={loads} fuse={fuse} pdl={pdl}: {us:8.1f} us a layer, "
                  f"{nbytes / us / 1e3:6.1f} GB/s{same}", flush=True)


if __name__ == "__main__":
    main()
