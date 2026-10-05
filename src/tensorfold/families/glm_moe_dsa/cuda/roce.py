"""Decode-window reductions over RoCE (b12x RoCEnante, in the serving image): one launch, every peer's partial
RDMA-written into pinned host memory, summed in fixed rank order - every rank stores the same bits, as the NCCL
all-gather + rank-order sum does - and capturable in CUDA graphs. Prompt chunks stay on NCCL (large messages)."""

from __future__ import annotations

import os

import torch

MAX_BYTES = 1 << 20              # decode windows: up to 32 rows x 6144 fp32 = 768 KB; logits argmax rows: tiny
HEALTH = os.environ.get("TF_GLM53_ROCE_HEALTH", "1") != "0"    # every window: all ranks agree no runtime timed out


def hca_names(value: str | None) -> list[str] | None:
    """NCCL_IB_HCA as RoCEnante device names: NCCL's exact-match "=" prefix and ":port" suffixes dropped (its proxy
    compares names verbatim, so "=rocep1s0f0" found no device). An exclusion list ("^...") names none: refused."""
    v = (value or "").strip()
    if v.startswith("^"):
        raise ValueError(f"NCCL_IB_HCA={value!r} excludes devices; RoCE reductions need the devices named")
    names = [n.split(":", 1)[0].strip() for n in v.lstrip("=").split(",")]
    return [n for n in names if n] or None


class RoceReduce:
    def __init__(self, rank: int, world: int, master: str, port: int, nccl=None) -> None:
        import torch.distributed as dist
        from b12x.comm.roce import AllReduce

        if os.environ.get("NCCL_SOCKET_IFNAME"):         # gloo would pick the hostname's loopback in a container
            os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ["NCCL_SOCKET_IFNAME"])
        if not dist.is_initialized():                    # the runtime's setup exchange only (gloo over TCP)
            from datetime import timedelta

            dist.init_process_group("gloo", init_method=f"tcp://{master}:{port}", rank=rank, world_size=world,
                                    timeout=timedelta(seconds=120))
        hca = hca_names(os.environ.get("NCCL_IB_HCA"))
        gid = os.environ.get("NCCL_IB_GID_INDEX")
        self.device = torch.device("cuda", 0)
        self.rt = AllReduce(exchange_group=dist.group.WORLD, device=self.device, max_size=MAX_BYTES,
                            max_gather_bytes=MAX_BYTES, hca_names=hca, gid_index=int(gid) if gid else None)
        self.rt.prepare((torch.float32,), padded_gather=True)
        self.rank, self.world, self.nccl = rank, world, nccl
        self._flag = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self._every = torch.zeros((world,), dtype=torch.int32, device=self.device)
        if nccl is not None:
            self._check(nccl)

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.rt.all_reduce(x, out=out)

    def all_gather(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.rt.all_gather(x.reshape(-1), dim=0, out=out.view(-1))

    def healthy_everywhere(self) -> None:
        """After a window's reductions, before its tokens are read: every rank agrees that no RoCE runtime timed out.
        b12x is fail-stop (a wait past B12X_ROCE_SPIN_LIMIT poisons the runtime and later launches do nothing), so
        inside a CUDA graph a timeout would leave stale sums and emit wrong tokens; every rank raises instead."""
        self._flag.fill_(1 if self.rt.poisoned else 0)
        if self.nccl is None:
            bad = [self.rank] if int(self._flag.item()) else []
        else:
            self.nccl.all_gather(self._flag, self._every)
            bad = [r for r, v in enumerate(self._every.tolist()) if v]
        if not bad:
            return
        if self.rt.poisoned:
            self.rt.check_health()                       # b12x's own reason, on the rank that timed out
        raise RuntimeError(f"RoCE reductions failed on rank {', '.join(map(str, bad))}: tokens of this window dropped")

    def _check(self, nccl) -> None:
        """Every rank's sum must carry the same bits (compared through NCCL), and the bits of the NCCL fallback's
        rank-order sum, else refuse to serve on RoCE (every rank reaches the same verdict, so every rank falls back
        to NCCL); checked at a 3-row window and the widest decode window (fused.DECODE_ROWS)."""
        from .fused import DECODE_ROWS

        same = order = True
        for rows in (3, DECODE_ROWS):
            g = torch.Generator(device="cpu").manual_seed(1000 + self.rank + rows)
            x = (torch.randn(rows, 6144, generator=g) * 10.0 ** (self.rank - 1)).to(self.device)
            s = torch.empty_like(x)
            self.all_reduce(x, s)
            allv = torch.empty((self.world, rows, 6144), device=self.device)
            nccl.all_gather(s, allv)
            parts = torch.empty((self.world, rows, 6144), device=self.device)
            nccl.all_gather(x, parts)
            ref = parts[0].clone()
            for r in range(1, self.world):
                ref += parts[r]
            same &= all(torch.equal(allv[r], allv[0]) for r in range(self.world))
            order &= torch.equal(allv[0], ref)
        if not same:
            raise RuntimeError("RoCE all-reduce: ranks hold different bits")
        if not order:
            raise RuntimeError("RoCE all-reduce: the sum is not the NCCL rank-order sum (decode would change bits)")
        print(f"[tensorfold] RoCE one-shot reduce ready (ranks bit-equal: {same}; equals the NCCL rank-order sum: "
              f"{order}; windows of 3 and {DECODE_ROWS} rows)", flush=True)
