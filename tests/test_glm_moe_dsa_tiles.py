"""Full GLM-5.3's saved tile tables (TF_GLM53_TILES) on a CPU, no weights: a saved table restores every linear's
tile, a table for other linears is refused before any tile changes, and a load replaces the boot's timing."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3.linear import Exl3Linear  # noqa: E402
from tensorfold.families.glm_moe_dsa.cuda import tiles  # noqa: E402

SHAPES = [(6144, 2048, 3.0, "mcg"), (2048, 4096, 3.0, "mcg"), (2048, 4096, 4.0, "mcg"), (512, 6144, 2.0, "mul1")]


def _lins():
    """Exl3Linear objects with GLM-5.3's dense shapes (no words: tiles are all a table holds)."""
    z = torch.zeros(1)
    return [Exl3Linear(z, z, z, None, bits, cb, k, n) for k, n, bits, cb in SHAPES]


def test_a_saved_table_restores_every_tile(tmp_path):
    a = _lins()
    picks = [(4, 8), (1, 8), (8, 4), (1, 2)]
    for x, t in zip(a, picks):
        x.split = t
    sha = tiles.save(a, str(tmp_path / "t" / "tiles.json"))
    b = _lins()
    got, changed = tiles.load(b, str(tmp_path / "t" / "tiles.json"))
    assert [x.split for x in b] == picks
    assert got == sha == tiles.digest(a) == tiles.digest(b)
    assert changed == sum(p != x.split for p, x in zip(picks, _lins()))


def test_the_sweep_tables_load_and_hash_as_written(tmp_path):
    """Tables written before this module (the same JSON, sort_keys, no trailing newline) load, and the logged sha is
    the sha256 of the file's bytes."""
    table = [[k, n, bits, cb, "strips", 2, 8] for k, n, bits, cb in SHAPES]
    blob = json.dumps({"count": len(table), "linears": table}, sort_keys=True)
    path = tmp_path / "tiles.json"
    path.write_bytes(blob.encode())
    lins = _lins()
    sha, _ = tiles.load(lins, str(path))
    assert sha == hashlib.sha256(blob.encode()).hexdigest() == tiles.digest(lins)


@pytest.mark.parametrize("edit", ["count", "k", "bits", "codebook", "layout"])
def test_a_table_for_other_linears_is_refused_untouched(tmp_path, edit):
    a = _lins()
    for x in a:                                          # tiles that differ from the plan everywhere, so a table
        x.split = (x.split[0] * 2, 2)                    # taken in part would show
    path = str(tmp_path / "tiles.json")
    tiles.save(a, path)
    data = json.loads(open(path).read())
    i = {"k": 0, "bits": 2, "codebook": 3, "layout": 4}.get(edit)
    if edit == "count":
        data["linears"].append(list(data["linears"][0]))
    else:
        row = data["linears"][-1]
        row[i] = {"k": 4096, "bits": 4.0, "codebook": "3inst", "layout": "stored"}[edit]
    open(path, "w").write(json.dumps(data))
    b = _lins()
    before = [x.split for x in b]
    with pytest.raises(ValueError, match="refusing it"):
        tiles.load(b, path)
    assert [x.split for x in b] == before


def test_spec():
    assert tiles.spec("") == ("", "")
    assert tiles.spec("save:/x/tiles.json") == ("save", "/x/tiles.json")
    assert tiles.spec("load:C:/t.json") == ("load", "C:/t.json")
    for bad in ("tiles.json", "load:", "keep:/x"):
        with pytest.raises(ValueError, match="TF_GLM53_TILES"):
            tiles.spec(bad)


def test_after_load_saves_what_runs(tmp_path, monkeypatch, capsys):
    lins = _lins()
    path = tmp_path / "saved.json"
    monkeypatch.setenv("TF_GLM53_TILES", f"save:{path}")
    tiles.after_load(SimpleNamespace(tunable=lins), 2)
    assert path.exists() and tiles.load(_lins(), str(path))[0] == tiles.digest(lins)
    assert f"rank 2: tiles in use sha {tiles.digest(lins)[:16]}" in capsys.readouterr().out


def test_weights_load_a_table_instead_of_timing(tmp_path, monkeypatch):
    """fused.Weights with TF_GLM53_TILES=load:PATH runs the saved tiles and never times any (no tune_* call)."""
    pytest.importorskip("triton")
    from tensorfold.families.glm_moe_dsa.cuda import fused
    from tensorfold.families.glm_moe_dsa.cuda.weights import Layer

    def lin(k, n):
        return Exl3Linear(torch.zeros(1), torch.zeros(1), torch.zeros(1), None, 3.0, "mcg", k, n)

    cfg = SimpleNamespace(num_attention_heads=8, qk_rope_head_dim=4, rope_theta=10000.0, qk_nope_head_dim=4,
                          v_head_dim=4, kv_lora_rank=8)
    heads = cfg.num_attention_heads // 4

    def layer(i):
        return Layer(i, None, None, lin(256, 128), None, lin(256, 128), None, lin(128, 256),
                     torch.zeros((heads * 8, 8), dtype=torch.bfloat16), lin(128, 256), None, None, None,
                     {"gate": lin(256, 384), "up": lin(256, 384), "down": lin(384, 256)})

    saved = [layer(i) for i in range(2)]
    every = [x for L in saved for x in fused.linears(L)]
    for j, x in enumerate(every):
        x.split = (1 + j % 2, 4)
    path = str(tmp_path / "tiles.json")
    tiles.save(every, path)

    def no_timing(*a, **k):
        raise AssertionError("timed tiles although a table was loaded")

    monkeypatch.setattr(fused, "tune_linears", no_timing)
    monkeypatch.setattr(fused, "tune_groups", no_timing)
    monkeypatch.setattr(fused, "TUNE", True)
    monkeypatch.setenv("TF_GLM53_TILES", f"load:{path}")
    w = fused.Weights(cfg, 0, 4, None, torch.zeros((4, 8)), None, torch.zeros((4, 8)), [layer(i) for i in range(2)],
                      None)
    assert [x.split for x in w.tunable] == [x.split for x in every]
