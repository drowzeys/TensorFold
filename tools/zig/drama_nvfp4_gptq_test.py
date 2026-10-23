"""drama_nvfp4_gptq.py on the CPU (EQ_GPTQ_TEST_DEVICE=cuda: the CLI on the GPU): C2/Zig bytes, oracles, refusals."""
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# the GPU may host a live server: in-process tests always hide it; only EQ_GPTQ_TEST_DEVICE=cuda shows it to the CLI
DEVICE = os.environ.get("EQ_GPTQ_TEST_DEVICE", "cpu")
if DEVICE not in ("cpu", "cuda"):
    sys.exit(f"EQ_GPTQ_TEST_DEVICE: cpu (default) or cuda, not {DEVICE!r}")
CLI_ENV = dict(os.environ) if DEVICE == "cuda" else dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1")
os.environ.update(CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1")
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import drama_nvfp4_gptq as q  # noqa: E402

if torch.cuda.is_initialized():
    sys.exit("drama_nvfp4_gptq_test: CUDA was initialised before this module hid it; run it in its own process")

TOOL = Path(__file__).with_name("drama_nvfp4_gptq.py")
SHARD = "model-00001-of-00001.safetensors"
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# nibbles, scale codes, f32 global bits (hex) of exp/eq-nvfp4-quant 2382e5d2 nvfp4.zig `encode` on golden_inputs()
ZIG_GOLDEN = (
    (("276de0a54e7cd3166ee9b53f6cd2065efbc4377de0a65f7cd3176ef9b54f7de2",
      "a1c3e597b2d4f6c6a1c3e597b2d4f6c6a1c3e597b2d4f6c690b2d406a1c3e5a5",
      "00000000000000000000001111111111734cbff5602ed7f45b9fe6713dc7f56a",
      "90b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f6"),
     "7e7e7c7c7c75696000017c7c70707070", "3ab46b47"),
    (("d7377ef0a65f7ce3166ee9b54f7de2065efac4276de0a54e7bd3166ee9b54f6c",
      "a1c3e597b2d4f6c6a1c3e597b2d4f6c6a1c3e597b2d4f6c690b2d406a1c3e5a5",
      "00000000000000001132445565667677d6724cafe5602dc6f35b9ee6613dc7e4",
      "90b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f6"),
     "7b7d7e7e7c75696000177e7e70707070", "36998a9f"),
    (("d7065efac4376de1a65ffcd4276ef9b64f7cd2065eeac4377de1965ffbd4276e",
      "a1c3e597b2d4f6c6a1c3e597b2d4f6c6a1c3e597b2d4f6c690b2d406a1c3e5a5",
      "00000000000000000000000000000000c7f45a0ed6624cbff5691ed7735cafe6",
      "90b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f690b2d4f6"),
     "7d7b7e7c7c75696000017e7c70707070", "3de49249"),
)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def e4m3(code):
    """C1's e4m3fn value of a scale code: 4 exponent bits biased by 7, 3 mantissa bits, subnormal at exponent 0."""
    exp, man = code >> 3, code & 7
    return man / 8 * 2.0**-6 if exp == 0 else (1 + man / 8) * 2.0 ** (exp - 7)


def nearest(v, s, g):
    """The e2m1 index whose reconstruction (q s) g is nearest |v|, the even index on an exact tie."""
    return min(range(8), key=lambda j: (abs(abs(v) - (E2M1[j] * s) * g), j % 2))


def loss(block, h, s, g):
    total = 0.0
    for v, hc in zip(block, h):
        rec = (E2M1[nearest(v, s, g)] * s) * g
        d = v - (-rec if v < 0 else rec)
        total += hc * d * d
    return total


def nibble(v, code, g):
    j = nearest(v, e4m3(code), g) if code else 0
    return j | 8 if v < 0 and j else j


def pack(nibbles):
    return [nibbles[i] | nibbles[i + 1] << 4 for i in range(0, len(nibbles), 2)]


def losses(block, h, g):
    """C2 step 4's loss of every scale code 1..126 for one 16-block."""
    return [loss(block, h, e4m3(code), g) for code in range(1, 127)]


def pick(block, h, g):
    """C2 step 4 from the contract text: the code of least loss, the smaller on ties; 0 for an all-zero block."""
    if not any(block):
        return 0
    ls = losses(block, h, g)
    return min(range(1, 127), key=lambda c: (ls[c - 1], c))


def c2(w, h):
    """C2 steps 3 and 4 for W [n][k] (floats) with importance h [k]: g, scale codes, packed nibbles."""
    amax = max(abs(v) for row in w for v in row)
    g = f32(amax / (6 * 448)) if amax else 1.0
    scales, nibbles = [], []
    for row in w:
        for b in range(0, len(row), 16):
            code = pick(row[b:b + 16], h[b:b + 16], g)
            scales.append(code)
            nibbles += [nibble(v, code, g) for v in row[b:b + 16]]
    return g, scales, pack(nibbles)


def sequential_gptq(w, h, g, u):
    """f64 GPTQ of W [n][k] by columns: codes, nibbles, codes picked on the unfed W, the closest decision's margin."""
    w, w0 = w.clone(), w
    n, k = w.shape
    codes, unfed, nibbles, margin = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)], math.inf
    for c in range(k):
        if c % 16 == 0:
            for r in range(n):
                ls = sorted(losses(w[r, c:c + 16].tolist(), h[c:c + 16], g))
                margin = min(margin, (ls[1] - ls[0]) / ls[1])
                codes[r].append(pick(w[r, c:c + 16].tolist(), h[c:c + 16], g))
                unfed[r].append(pick(w0[r, c:c + 16].tolist(), h[c:c + 16], g))
        qc = torch.empty(n, dtype=torch.float64)
        for r in range(n):
            v, s = w[r, c].item(), e4m3(codes[r][-1])
            margin = min(margin, *(abs(abs(v) - (m * s) * g) / (s * g) for m in MIDPOINTS))
            nibbles[r].append(nibble(v, codes[r][-1], g))
            qc[r] = math.copysign((E2M1[nibbles[r][-1] & 7] * s) * g, v)
        err = (w[:, c] - qc) / u[c, c]
        w[:, c + 1:] -= err[:, None] * u[c, c + 1:]
    return codes, pack(sum(nibbles, [])), unfed, margin


def golden_inputs():
    """Three W [4][64]: generic values, exact midpoints (m s) g and their f64 neighbours, zero and tiny blocks; h."""
    h = [2.0 ** (c * 5 % 13 - 6) for c in range(64)]
    h[7], h[40] = 1e-9, 1e3
    mats = []
    for amax in (3.7, 0.0123, 300.0):
        g = f32(amax / 2688)
        generic = [amax * 1.9 * (((c + int(amax * 1000)) * 7919 % 97) / 97 - 0.5) for c in range(64)]
        generic[0] = amax
        edges = []
        for code in (0x7E, 0x78, 0x6B, 0x5D):
            s = e4m3(code)
            mids = [(m * s) * g for m in MIDPOINTS]
            block = mids + [math.nextafter(t, 0) for t in mids[:4]] + [math.nextafter(t, math.inf) for t in mids[4:]]
            edges += [v * (-1) ** i for i, v in enumerate(block + [(4 * s) * g, (1.5 * s) * g])]
        tiny = [0.0] * 16 + [1e-7 * (i + 1) for i in range(16)] + [-v for v in generic[32:]]
        exact = [(E2M1[i % 8] * e4m3(0x70)) * g * (-1) ** i for i in range(64)]
        mats.append([generic, edges, tiny, exact])
    return mats, h


def correlated(rows, width, rank, gen):
    """bf16 rows of low effective rank with lognormal channel gains, the shape of real MoE inputs."""
    low = torch.randn(rows, rank, generator=gen) @ torch.randn(rank, width, generator=gen)
    return ((low + 0.1 * torch.randn(rows, width, generator=gen)) * torch.randn(width, generator=gen).exp()).bfloat16()


def skewed_picks(rows, gen):
    """8 distinct experts a row by a power law, so some experts are rare; expert 255 is never routed."""
    weights = 1 / torch.arange(1, q.EXPERTS + 1, dtype=torch.float64) ** 1.2
    weights[-1] = 0
    return torch.multinomial(weights.expand(rows, -1), q.TOP_K, generator=gen)


def write_checkpoint(folder, layer, inner, hidden, gen):
    """A one-shard FP8 checkpoint of one layer's 256 routed experts; returns its config.json sha256."""
    folder.mkdir()
    tensors = {}
    for e in range(q.EXPERTS):
        for proj, (n, k) in zip(q.PROJECTIONS, ((inner, hidden), (inner, hidden), (hidden, inner))):
            name = f"{q.PREFIX}{layer}.mlp.experts.{e}.{proj}."
            tensors[name + "weight"] = (torch.randn(n, k, generator=gen) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
            tensors[name + "weight_scale_inv"] = torch.rand(-(-n // 128), -(-k // 128), generator=gen) * 1e-4 + 1e-5
    save_file(tensors, folder / SHARD)
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, SHARD)}))
    (folder / "config.json").write_text('{"model_type": "fish_omni"}\n')
    return hashlib.sha256((folder / "config.json").read_bytes()).hexdigest()


def write_rows(folder, layer, x, picks, config):
    """C3 --rows files of one layer (every row prompt text) and the rows.json the probe writes beside them."""
    folder.mkdir(exist_ok=True)
    (folder / "class.u8").write_bytes(bytes(len(x)))
    x.view(torch.int16).numpy().tofile(folder / f"layer-{layer:02d}.x.bf16")
    picks.int().numpy().tofile(folder / f"layer-{layer:02d}.picks.i32")
    rows = len(x)
    (folder / "rows.json").write_text(json.dumps({
        "schema": "drama-probe-rows/1", "identity": {"config_sha256": config}, "rows": rows, "layers": q.LAYERS,
        "class": {"file": "class.u8", "dtype": "u8", "shape": [rows]},
        "x": {"file": "layer-LL.x.bf16", "dtype": "bf16", "shape": [rows, x.shape[1]]},
        "picks": {"file": "layer-LL.picks.i32", "dtype": "i32", "shape": [rows, q.TOP_K]}}))


def poke(path, offset, data):
    raw = bytearray(path.read_bytes())
    raw[offset:offset + len(data)] = data
    path.write_bytes(bytes(raw))


class QuantiseTests(unittest.TestCase):
    def test_a_diagonal_hessian_gives_the_c2_bytes(self):
        gen = torch.Generator().manual_seed(1)
        w = torch.randn(3, 16, 64, generator=gen, dtype=torch.float64) * torch.randn(64, generator=gen).exp()
        w[0, 0, :16] = 0
        w[0, 1, 16:32] *= 1e-6
        w[1, 2, 7] = 50
        h = (2 * torch.randn(64, generator=gen)).exp()
        h[7] = 1e-9
        damped = q.damp(torch.diag_embed(h).expand(3, -1, -1))
        hd = damped.diagonal(dim1=1, dim2=2).double()
        g = q.global_scale(w, "w")
        nib, codes, _ = q.quantise(w, hd, g[:, None].expand(-1, 16), q.upper(damped))
        for e in range(3):
            ref_g, scales, packed = c2(w[e].tolist(), hd[e].tolist())
            self.assertEqual(g[e].item(), ref_g)
            self.assertEqual(codes[e].flatten().tolist(), scales)
            self.assertEqual(nib[e].flatten().tolist(), packed)
        self.assertEqual(codes[0, 0, 0].item(), 0)

    def test_a_diagonal_hessian_gives_the_zig_encoders_bytes(self):
        mats, h = golden_inputs()
        self.assertEqual(len(ZIG_GOLDEN), len(mats))
        hd = torch.tensor(h, dtype=torch.float64)[None]
        feedback = q.upper(torch.diag_embed(hd.float()))
        for w, (nibbles, scales, global_bits) in zip(mats, ZIG_GOLDEN):
            w = torch.tensor(w, dtype=torch.float64)[None]
            g = q.global_scale(w, "golden")
            nib, codes, _ = q.quantise(w, hd, g[:, None].expand(-1, 4), feedback)
            self.assertEqual(nib.flatten().numpy().tobytes().hex(), "".join(nibbles))
            self.assertEqual(codes.flatten().numpy().tobytes().hex(), scales)
            self.assertEqual(struct.pack(">f", g.item()).hex(), global_bits)

    def test_exact_midpoints_round_to_the_even_index_and_magnitudes_saturate_at_6(self):
        a = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, 0.2, 0.3], dtype=torch.float64)
        one = torch.ones_like(a)
        self.assertEqual(q.nearest(a, one, one).tolist(), [0, 2, 2, 4, 4, 6, 6, 7, 0, 1])

    def test_b_gptq_beats_rtn_on_correlated_rows(self):
        gen = torch.Generator().manual_seed(2)
        x = correlated(2048, 128, 8, gen).float()
        hess = (q.gram(x) / len(x)).expand(2, -1, -1)
        w = torch.randn(2, 32, 128, generator=gen, dtype=torch.float64)
        (_, _, gptq), rtn = q.encode(w, hess, q.global_scale(w, "w")[:, None].expand(-1, 32))
        err, base = q.output_error(w, gptq, hess), q.output_error(w, rtn, hess)
        self.assertTrue(all(a < b for a, b in zip(err, base)), f"GPTQ {err} against RTN {base}")

    def test_lazy_batched_gptq_follows_a_sequential_oracle_that_picks_scales_after_the_feedback(self):
        gen = torch.Generator().manual_seed(5)
        x = correlated(1024, 384, 12, gen).float()
        damped = q.damp((q.gram(x) / len(x))[None])
        u, h = q.upper(damped), damped.diagonal(dim1=1, dim2=2).double()
        w = torch.randn(1, 4, 384, generator=gen, dtype=torch.float64)
        g = q.global_scale(w, "w")
        nib, codes, _ = q.quantise(w, h, g[:, None].expand(-1, 4), u)
        ref_codes, ref_nib, unfed, margin = sequential_gptq(w[0], h[0].tolist(), g.item(), u[0].double())
        # the tool's f32 cross-batch update errs far below 1e-5; no oracle decision (loss gap, midpoint gap) is closer
        self.assertGreater(margin, 1e-5)
        self.assertEqual(codes[0].tolist(), ref_codes)
        self.assertEqual(nib[0].flatten().tolist(), ref_nib)
        self.assertNotEqual(unfed, ref_codes)

    def test_a_tiny_nonzero_source_whose_global_rounds_to_zero_is_refused(self):
        code = torch.tensor([[1]], dtype=torch.uint8).view(torch.float8_e4m3fn)
        w = q.decode("tiny.", code, torch.tensor([[2.0**-149]]))
        self.assertEqual(w.item(), 2.7369110631344083e-48)
        with self.assertRaisesRegex(ValueError, "tiny expert 0: .* global 0.0, not a positive finite one"):
            q.global_scale(w[None], "tiny")
        self.assertEqual(q.global_scale(torch.zeros(1, 2, 16, dtype=torch.float64), "zero").tolist(), [1.0])

    def test_bad_sources_are_refused(self):
        codes, scale = torch.zeros(4, 32, dtype=torch.uint8), torch.ones(1, 1)
        nan = codes.clone()
        nan[1, 3] = 0x7F
        for c, s in ((nan, scale), (codes, scale * 0), (codes, scale * float("inf")), (codes, torch.ones(2, 1))):
            with self.assertRaises(ValueError):
                q.decode("x.", c.view(torch.float8_e4m3fn), s)

    def test_output_errors_report_0_over_0_as_0_nonzero_over_0_as_undefined_and_refuse_non_finite_sums(self):
        w, hess = torch.zeros(2, 2, 16, dtype=torch.float64), torch.eye(16)
        qw = w.clone()
        qw[1, 0, 3] = 0.5
        self.assertEqual(q.output_error(w, qw, hess.expand(2, -1, -1)), [0.0, "undefined"])
        w[:, :, 0] = 1
        self.assertEqual(q.output_error(w, w, hess.expand(2, -1, -1)), [0.0, 0.0])
        with self.assertRaises(ValueError):
            q.output_error(w, qw, (hess * float("inf")).expand(2, -1, -1))


class HessianTests(unittest.TestCase):
    def test_each_expert_sums_the_outer_products_of_exactly_its_rows(self):
        rows = 10
        x = torch.tensor([[r + 1, -(r % 3), 2 * r - 5, 7 - r] for r in range(rows)], dtype=torch.float32)
        members = {3: [0, 4, 9], 17: list(range(rows)), 200: [1, 2], 250: [5]}
        picks = []
        for r in range(rows):
            mine = [e for e, m in members.items() if r in m]
            slots = mine + [20 + 8 * r + j for j in range(q.TOP_K - len(mine))]
            picks.append(slots[r % q.TOP_K:] + slots[:r % q.TOP_K])
            for j in range(q.TOP_K - len(mine)):
                members[20 + 8 * r + j] = [r]
        hess, count, got, pool = q.gate_up_hessians(x, torch.tensor(picks))
        for e in range(q.EXPERTS):
            expected = sum((torch.outer(x[r], x[r]) for r in members.get(e, [])), torch.zeros(4, 4))
            self.assertTrue(torch.equal(hess[e], expected), e)
            self.assertEqual(count[e].item(), len(members.get(e, [])))
            self.assertEqual(got[e].tolist(), members.get(e, []))
        self.assertTrue(torch.equal(pool, sum(torch.outer(v, v) for v in x) / rows))

    def test_non_finite_hessians_and_activations_are_refused(self):
        picks = torch.arange(16).reshape(2, 8)
        with self.assertRaisesRegex(ValueError, "gate/up Hessian"):
            q.gate_up_hessians(torch.full((2, 4), 1e30), picks)
        _, _, members, _ = q.gate_up_hessians(torch.full((2, 4), 10.0), picks)
        big = torch.zeros(q.EXPERTS, 2, 4)
        big[0] = 1e38
        with self.assertRaisesRegex(ValueError, "expert 0: down activation"):
            q.down_hessians(torch.full((2, 4), 10.0), members, big, big)


class RowsTests(unittest.TestCase):
    def test_well_formed_rows_load_and_malformed_ones_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            root, gen = Path(folder), torch.Generator().manual_seed(4)
            x, picks = correlated(6, 32, 2, gen), skewed_picks(6, gen)

            def manifest(**change):
                doc = json.loads((root / "rows.json").read_text())
                doc.update(change)
                (root / "rows.json").write_text(json.dumps(doc))

            bad = {
                "nan input": lambda: poke(root / "layer-00.x.bf16", 2, b"\xc0\x7f"),
                "class 3": lambda: poke(root / "class.u8", 5, b"\x03"),
                "expert twice": lambda: poke(root / "layer-00.picks.i32", 4, bytes(picks[0, :1].int().numpy())),
                "expert 256": lambda: poke(root / "layer-00.picks.i32", 0, (256).to_bytes(4, "little")),
                "short x": lambda: (root / "layer-00.x.bf16").write_bytes((root / "layer-00.x.bf16").read_bytes()[:-2]),
                "other checkpoint": lambda: manifest(identity={"config_sha256": "0" * 64}),
                "row count": lambda: manifest(rows=7),
                "x dtype": lambda: manifest(x={"file": "layer-LL.x.bf16", "dtype": "f16", "shape": [6, 32]}),
                "layer count": lambda: manifest(layers=0),
                "schema": lambda: manifest(schema="drama-probe-rows/2"),
            }
            write_rows(root, 0, x, picks, "c0ffee")
            got_x, got_picks = q.routed(root, 0, 32, "c0ffee")
            self.assertTrue(torch.equal(got_x, x) and torch.equal(got_picks, picks))
            for name, corrupt in bad.items():
                with self.subTest(name):
                    write_rows(root, 0, x, picks, "c0ffee")
                    corrupt()
                    with self.assertRaises(ValueError):
                        q.routed(root, 0, 32, "c0ffee")


class ToolTests(unittest.TestCase):
    LAYER, INNER, HIDDEN, ROWS = 3, 16, 32, 512

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        gen = torch.Generator().manual_seed(3)
        config = write_checkpoint(root / "ckpt", cls.LAYER, cls.INNER, cls.HIDDEN, gen)
        cls.picks = skewed_picks(cls.ROWS, gen)
        write_rows(root / "rows", cls.LAYER, correlated(cls.ROWS, cls.HIDDEN, 4, gen), cls.picks, config)
        command = [sys.executable, str(TOOL), str(root / "ckpt"), str(root / "rows")]
        cls.runs = [subprocess.run(command + [str(root / f"out{i}"), "--layers", str(cls.LAYER)], capture_output=True,
                                   text=True, check=True, env=CLI_ENV) for i in range(2)]
        cls.out = [root / f"out{i}" / f"layer-{cls.LAYER:02d}.safetensors" for i in range(2)]

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_c_the_file_holds_exactly_the_c1_tensors_and_the_metadata(self):
        expected = {}
        for e in range(q.EXPERTS):
            for proj, (n, k) in zip(q.PROJECTIONS, ((self.INNER, self.HIDDEN),) * 2 + ((self.HIDDEN, self.INNER),)):
                name = f"model.language_model.layers.{self.LAYER}.mlp.experts.{e}.{proj}."
                expected |= {name + "weight": ("U8", [n, k // 2]), name + "weight_scale": ("F8_E4M3", [n, k // 16]),
                             name + "weight_scale_2": ("F32", [])}
        with safe_open(self.out[0], "pt") as f:
            self.assertEqual(set(f.keys()), set(expected))
            for name, kind in expected.items():
                self.assertEqual((f.get_slice(name).get_dtype(), f.get_slice(name).get_shape()), kind)
                if name.endswith("weight_scale"):
                    self.assertFalse(((f.get_tensor(name).view(torch.uint8) & 0x7F) == 0x7F).any())
            meta = json.loads(f.metadata()["gptq"])
        self.assertEqual(meta["rows"], torch.bincount(self.picks.flatten(), minlength=256).tolist())
        self.assertEqual((meta["kappa"], meta["damping"], sorted(meta["prior"])), (256, 0.01, ["down", "gate_up"]))
        errors = meta["output_error"]
        self.assertEqual(set(errors), {name for name in expected if name.endswith(".weight")})
        for proj in q.PROJECTIONS:
            pairs = [v for name, v in errors.items() if proj in name]
            gptq, rtn = (sum(p[key] for p in pairs) / len(pairs) for key in ("gptq", "rtn"))
            self.assertLess(gptq, rtn, proj)
        self.assertIn(f"{self.ROWS} rows, mean output error GPTQ/RTN gate_proj", self.runs[0].stdout)

    def test_d_two_runs_write_identical_bytes(self):
        self.assertEqual(self.out[0].read_bytes(), self.out[1].read_bytes())

    def test_a_layer_outside_the_model_is_refused(self):
        bad = subprocess.run([sys.executable, str(TOOL), "a", "b", "c", "--layers", "40"], capture_output=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
