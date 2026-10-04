"""Decode-window reductions over RoCE (b12x RoCEnante, in the serving image): one launch, every peer's partial
RDMA-written into pinned host memory, summed in fixed rank order - every rank stores the same bits, as the NCCL
all-gather + rank-order sum does - and capturable in CUDA graphs. Prompt chunks stay on NCCL (large messages).

Rails: a DGX Spark's QSFP port reaches the GB10 over two PCIe Gen5 x4 links, so one cabled port is two RoCE devices
("twins", rocep1s0f1 and roceP2p1s0f1, each ~112 Gb/s; MiaAI-Lab GLM-5.3-Flash v1.3.3 #30). The devices come from
TF_GLM53_ROCE_HCA, else NCCL_IB_HCA - a comma list, in the same rail order on every rank (rail i = one subnet on every
node: tools/tp4_run.sh orders them by subnet). Their RoCE v2 GID index: TF_GLM53_ROCE_GIDS (one a device), else
NCCL_IB_GID_INDEX for all; b12x takes one index for every device, so devices whose GIDs differ fall back to the first
device. TF_GLM53_RAILS=1: the first device only."""

from __future__ import annotations

import os

import torch

MAX_BYTES = 1 << 20              # decode windows: up to 32 rows x 6144 fp32 = 768 KB; logits argmax rows: tiny


def rails() -> tuple[list[str] | None, int | None]:
    """(RoCE device names in rail order, or None: b12x discovers them; their one RoCE v2 GID index, or None)."""
    spec = os.environ.get("TF_GLM53_ROCE_HCA") or os.environ.get("NCCL_IB_HCA") or ""
    if not spec or spec.startswith("^"):              # unset, or an NCCL exclusion list: b12x's own discovery
        names = None
    else:                                             # NCCL syntax: "=" exact-match prefix, ":port" suffixes
        names = [h.split(":")[0] for h in spec.lstrip("=").split(",") if h.strip()] or None
    gids = [int(g) for g in os.environ.get("TF_GLM53_ROCE_GIDS", "").split(",") if g.strip()]
    one = os.environ.get("NCCL_IB_GID_INDEX")
    if names and gids and len(gids) != len(names):
        raise ValueError(f"TF_GLM53_ROCE_GIDS={gids}: one GID index for each of {names}")
    if names and gids and len(set(gids)) > 1:        # b12x takes one GID index for all its devices
        print(f"[tensorfold] RoCE devices {names} have different GID indices {gids}: using {names[0]} only",
              flush=True)
        names, gids = names[:1], gids[:1]
    if names and os.environ.get("TF_GLM53_RAILS", "2") == "1":
        names = names[:1]
    gid = gids[0] if gids else (int(one) if one else None)
    return names, gid


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
        names, gid = rails()
        self.rt = AllReduce(exchange_group=dist.group.WORLD, device=torch.device("cuda", 0), max_size=MAX_BYTES,
                            max_gather_bytes=MAX_BYTES, hca_names=names, gid_index=gid)
        self.hcas = names
        self.rt.prepare((torch.float32,), padded_gather=True)
        self.rank, self.world = rank, world
        if nccl is not None:
            self._check(nccl)

    def all_reduce(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.rt.all_reduce(x, out=out)

    def all_gather(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return self.rt.all_gather(x.reshape(-1), dim=0, out=out.view(-1))

    def _check(self, nccl) -> None:
        """Every rank's sum must carry the same bits (compared through NCCL), else refuse to serve on RoCE; checked
        at a 3-row window and the widest decode window (fused.DECODE_ROWS: concurrent DFlash2 rounds)."""
        from .fused import DECODE_ROWS

        same = order = True
        for rows in (3, DECODE_ROWS):
            g = torch.Generator(device="cpu").manual_seed(1000 + self.rank + rows)
            x = (torch.randn(rows, 6144, generator=g) * 10.0 ** (self.rank - 1)).cuda()
            s = torch.empty_like(x)
            self.all_reduce(x, s)
            allv = torch.empty((self.world, rows, 6144), device="cuda")
            nccl.all_gather(s, allv)
            parts = torch.empty((self.world, rows, 6144), device="cuda")
            nccl.all_gather(x, parts)
            ref = parts[0].clone()
            for r in range(1, self.world):
                ref += parts[r]
            same &= all(torch.equal(allv[r], allv[0]) for r in range(self.world))
            order &= torch.equal(allv[0], ref)
        if not same:
            raise RuntimeError("RoCE all-reduce: ranks hold different bits")
        print(f"[tensorfold] RoCE one-shot reduce ready on {','.join(self.hcas or ['(discovered)'])} (ranks "
              f"bit-equal: {same}; equals the NCCL rank-order sum: {order}; windows of 3 and {DECODE_ROWS} rows)",
              flush=True)
