"""Full GLM-5.3 from an EXL3 pack that stores the unquantized kv_b / router / indexer tensors as fp16 (CPU, no
weights): they reach the bf16 kernels as the same values, and a tensor bf16 cannot hold exactly is refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.glm_moe_dsa.cuda import weights  # noqa: E402


def test_exact_fp16_values_become_the_same_bf16_values():
    src = (torch.randn(64, 32) * 3).to(torch.bfloat16)           # bf16 originals, stored by the pack as fp16
    got = weights.as_bf16(src.to(torch.float16), "w")
    assert got.dtype == torch.bfloat16 and torch.equal(got, src)


@pytest.mark.parametrize("value", [1.0 + 2.0 ** -9, 65504.0 - 32.0, float("nan")])
def test_values_bf16_cannot_hold_are_refused(value):
    t = torch.ones(8, dtype=torch.float16)
    t[3] = value
    with pytest.raises(ValueError, match="refusing a rounding cast"):
        weights.as_bf16(t, "model.layers.3.self_attn.kv_b_proj.weight")


def test_other_dtypes_pass_through_untouched():
    for t in (torch.ones(4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.float32)):
        assert weights.as_bf16(t, "w") is t


class _Reader:
    """Every plain tensor fp16, as the EXL3 pack stores them; EXL3 linears are not read (load_linear is faked)."""

    def __init__(self, bad: str = "") -> None:
        self.bad = bad

    def has(self, name: str) -> bool:
        return "indexer" in name

    def get(self, name: str, device="cpu") -> torch.Tensor:
        t = torch.full((4,), 0.5, dtype=torch.float16)
        if name == self.bad:
            t[0] = 1.0 + 2.0 ** -9                               # an fp16 value with more bits than bf16 keeps
        return t


def test_load_layer_hands_the_kernels_bf16(monkeypatch):
    monkeypatch.setattr(weights, "load_linear", lambda r, prefix, device="cuda": prefix)
    cfg = SimpleNamespace(first_k_dense_replace=3)
    L = weights.load_layer(_Reader(), cfg, 5, device="cpu", experts=False)
    plain = [L.input_norm, L.post_attn_norm, L.q_a_norm, L.kv_a_norm, L.kv_b, *L.router,
             L.indexer["wk"], L.indexer["weights_proj"], *L.indexer["k_norm"]]
    assert all(t.dtype == torch.bfloat16 for t in plain)
    with pytest.raises(ValueError, match="kv_b_proj"):
        weights.load_layer(_Reader("model.layers.5.self_attn.kv_b_proj.weight"), cfg, 5, device="cpu", experts=False)
