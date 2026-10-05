"""Full GLM-5.3 loading on a unified-memory GPU (CPU, no weights): the reader closes a layer's files and drops their
page cache before the next layer opens, and before the caches are sized."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tensorfold.families.glm_moe_dsa.cuda import weights  # noqa: E402


def _checkpoint(tmp_path, layers: int = 3):
    """One shard a layer plus an extra file outside the index (a separate lm_head file)."""
    where = {}
    for i in range(layers):
        name = f"model.layers.{i}.input_layernorm.weight"
        fn = f"model-{i:05d}.safetensors"
        safetensors_torch.save_file({name: torch.full((8,), float(i), dtype=torch.bfloat16)}, str(tmp_path / fn))
        where[name] = fn
    safetensors_torch.save_file({"lm_head.weight": torch.zeros((4, 8), dtype=torch.bfloat16)},
                                str(tmp_path / "lm_head.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": where}))
    return tmp_path


@pytest.fixture
def advised(monkeypatch):
    """The files posix_fadvise(DONTNEED) was called on (by path), instead of the real call."""
    import os

    seen: list[str] = []
    paths: dict[int, str] = {}
    real_open = os.open

    def fake_open(path, flags, *a):
        fd = real_open(path, flags, *a)
        paths[fd] = str(path)
        return fd

    def fake_fadvise(fd, offset, length, advice):
        assert advice == os.POSIX_FADV_DONTNEED and (offset, length) == (0, 0)
        seen.append(paths[fd])

    monkeypatch.setattr(weights.os, "open", fake_open)
    monkeypatch.setattr(weights.os, "posix_fadvise", fake_fadvise, raising=False)
    monkeypatch.setattr(weights.os, "POSIX_FADV_DONTNEED", getattr(os, "POSIX_FADV_DONTNEED", 4), raising=False)
    return seen


def test_release_closes_the_files_read_and_drops_only_their_pages(tmp_path, advised):
    ckpt = _checkpoint(tmp_path)
    r = weights.RankReader(ckpt, 0, 4)
    r.get("model.layers.1.input_layernorm.weight")
    assert list(r._open) == ["model-00001.safetensors"]
    assert r.release() == 1
    assert r._open == {}
    assert [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in advised] == ["model-00001.safetensors"]


def test_drop_page_cache_covers_every_checkpoint_file(tmp_path, advised):
    ckpt = _checkpoint(tmp_path)
    r = weights.RankReader(ckpt, 0, 4)
    r.get("model.layers.0.input_layernorm.weight")
    assert r.drop_page_cache() == 4                  # three indexed shards and the file outside the index
    assert r._open == {}
    names = sorted(p.replace("\\", "/").rsplit("/", 1)[-1] for p in advised)
    assert names == ["lm_head.safetensors", "model-00000.safetensors", "model-00001.safetensors",
                     "model-00002.safetensors"]


def test_layers_load_one_at_a_time(tmp_path, monkeypatch, advised):
    """When layer i loads, no file from layer i - 1 is still open (each layer's handles keep its pages mapped)."""
    ckpt = _checkpoint(tmp_path, layers=4)
    r = weights.RankReader(ckpt, 0, 4)
    open_at_start = []

    def fake_load_layer(reader, cfg, i, device="cuda", experts=True):
        open_at_start.append(len(reader._open))
        return reader.get(f"model.layers.{i}.input_layernorm.weight")

    monkeypatch.setattr(weights, "load_layer", fake_load_layer)
    got = weights.load_layers(r, cfg=None, n=4, device="cpu")
    assert [float(t[0]) for t in got] == [0.0, 1.0, 2.0, 3.0]
    assert open_at_start == [0, 0, 0, 0]
    assert r._open == {}
    assert len(advised) == 4
