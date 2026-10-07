"""FastH3: checkpoint mapping, the distilled schedule, and routed attention wired into the H3 transformer."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from tensorfold.families.h3 import fasth3
from tensorfold.families.h3.config import DiTConfig
from tensorfold.families.h3.dit import H3DiT
from tensorfold.families.h3.sampler import denoise
from tensorfold.families.h3.schedule import Schedule

SMALL = DiTConfig(hidden_size=32, num_layers=2, token_refiner_num_layers=1, num_attention_heads=2,
                  attention_head_dim=16, ffn_hidden_size=48, latents_dim=24, audio_latents_dim=32, text_dim=20,
                  timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=24, adaln_out_features=576,
                  final_adaln_out_features=64, rope_inv_freq_len=2)
BACK = (("blocks.", "transformer_blocks."), ("token_refiner.transformer_blocks.", "token_refiner.refiner_blocks."),
        ("time_embedder.proj_in", "time_embedder.linear_1"), ("time_embedder.proj_out", "time_embedder.linear_2"),
        ("audio_patch_proj", "audio_proj_in"), ("final_layer.audio_out", "audio_proj_out"),
        ("condition_proj", "context_embedder"), ("final_layer.adaln_proj.linear", "norm_out.linear"),
        ("final_layer.norm", "norm_out.norm"), ("video_patch_proj", "proj_in"), ("final_layer.video_out", "proj_out"),
        (".attn.out_proj", ".attn.to_out.0"), (".mlp.fc1", ".ff.net.0.proj"), (".mlp.fc2", ".ff.net.2"),
        (".attn.q_norm", ".attn.norm_q"), (".attn.k_norm", ".attn.norm_k"))


def small_model(seed=0):
    mx.random.seed(seed)
    model = H3DiT(SMALL)
    mx.eval(model.parameters())
    return model


def as_diffusers(model) -> dict:
    """The model's parameters under FastVideo's names and layouts."""

    heads, dim = SMALL.num_attention_heads, SMALL.attention_head_dim
    out = {}
    for name, value in tree_flatten(model.parameters()):
        for ours, theirs in BACK:
            name = name.replace(ours, theirs)
        if name.endswith(".attn.qkv_proj.weight"):
            rows = value.reshape(heads, 3, dim, value.shape[1])
            for slot, kind in enumerate(("to_q", "to_k", "to_v")):
                out[name.replace("qkv_proj", kind)] = rows[:, slot].reshape(heads * dim, value.shape[1])
        elif name.endswith(".ff.net.0.proj.weight"):
            half = value.shape[0] // 2
            out[name] = mx.concatenate([value[half:], value[:half]], axis=0)
        else:
            out[name] = value
    return out


def test_names_the_test_writes_are_the_ones_the_loader_reads():
    names = as_diffusers(small_model())
    assert "transformer_blocks.0.attn.to_q.weight" in names and "token_refiner.refiner_blocks.0.norm1.weight" in names
    assert "proj_in.weight" in names and "norm_out.linear.weight" in names and "audio_proj_out.bias" in names


def test_assemble_restores_fused_rows_and_swapped_halves():
    model = small_model(1)
    tensors = as_diffusers(model)
    tensors["transformer_blocks.1.attn.to_gate_compress.weight"] = mx.ones((32, 32))
    found, gates = fasth3.assemble(tensors, SMALL)
    expected = dict(tree_flatten(model.parameters()))
    assert found.keys() == expected.keys() and list(gates) == [1]
    for name, value in expected.items():
        np.testing.assert_array_equal(np.asarray(found[name]), np.asarray(value), err_msg=name)


def test_load_reads_a_checkpoint_folder_and_its_contract(tmp_path):
    model = small_model(2)
    (tmp_path / "transformer").mkdir()
    mx.save_safetensors(str(tmp_path / "transformer" / "diffusion_pytorch_model-00001-of-00001.safetensors"),
                        as_diffusers(model))
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({
        "num_attention_heads": 2, "attention_head_dim": 16, "hidden_size": 32, "num_layers": 2,
        "num_refiner_layers": 1, "ffn_dim": 48, "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
        "text_dim": 20, "freq_dim": 16, "time_embed_hidden_dim": 32, "time_embed_dim": 24, "rope_freq_dim": 2}))
    (tmp_path / fasth3.CONTRACT).write_text(json.dumps({
        "attention_backend": "VIDEO_SPARSE_ATTN_H3", "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "video_scheduler_shift": 10.0, "audio_scheduler_shift": 3.0, "vsa_sparsity": 0.8, "vsa_tile_size": 64,
        "task": "t2av"}))
    loaded, gates, rules = fasth3.load_fasth3(tmp_path)
    assert gates == {} and rules.forwards == 8 and rules.video_shift == 10.0 and rules.sparsity == 0.8
    assert rules.nodes[0] == pytest.approx(0.999) and rules.nodes[-1] == pytest.approx(0.125)
    for (name, a), (_, b) in zip(tree_flatten(loaded.parameters()), tree_flatten(model.parameters()), strict=True):
        np.testing.assert_array_equal(np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32)), err_msg=name)


def test_distilled_rungs_are_shifted_and_end_at_zero():
    nodes = (0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125)
    schedule = Schedule(10.0, 0, None, nodes)
    assert len(schedule) == 8 and schedule.sigmas[-1] == 0.0
    np.testing.assert_allclose(schedule.sigmas[:-1], [10 * s / (1 + 9 * s) for s in nodes], rtol=1e-6)
    np.testing.assert_allclose(schedule.timesteps, 1 - schedule.sigmas[:-1], rtol=1e-6)
    with pytest.raises(ValueError):
        Schedule(10.0, 0, None, (0.5, 0.75))


def run(model, text, **extra):
    return denoise(model, text, [1, 1, 1], 64, 64, 39, points=3, seed=4, **extra)


def test_routing_that_keeps_every_tile_equals_dense_attention():
    text = mx.random.normal((1, 3, SMALL.text_dim))
    dense = run(small_model(5), text)
    # no gates and sparsity 0: the routed call is the dense one, through the packed (rows, heads, dim) layout
    model = small_model(5)
    same = run(model, text, prepare=fasth3.route(model, {}, 0.0))
    np.testing.assert_allclose(np.asarray(same.video_rows), np.asarray(dense.video_rows), atol=1e-4, rtol=1e-4)
    # a sparsity so low that every video tile is kept: the gather path, which must still equal dense
    model = small_model(5)
    kept = run(model, text, prepare=fasth3.route(model, {}, 0.01))
    np.testing.assert_allclose(np.asarray(kept.video_rows), np.asarray(dense.video_rows), atol=2e-3, rtol=2e-3)
    np.testing.assert_allclose(np.asarray(kept.audio_rows), np.asarray(dense.audio_rows), atol=2e-3, rtol=2e-3)


def test_sparse_routing_with_gates_runs_and_differs():
    text = mx.random.normal((1, 3, SMALL.text_dim))
    dense = run(small_model(6), text)
    model = small_model(6)
    gates = {i: mx.random.normal((32, 32)) * 0.05 for i in range(SMALL.num_layers)}
    hook = fasth3.route(model, gates, 0.5)
    sparse = run(model, text, prepare=hook)
    assert np.isfinite(np.asarray(sparse.video_rows)).all() and np.isfinite(np.asarray(sparse.audio_rows)).all()
    assert not np.allclose(np.asarray(sparse.video_rows), np.asarray(dense.video_rows))
    geometry = hook(sparse.packed, sparse.latent_frames, sparse.latent_height, sparse.latent_width)
    assert geometry.total_seq_length == sparse.packed.rows and geometry.num_video_tiles >= 2
