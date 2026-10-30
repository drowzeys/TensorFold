#!/usr/bin/env python3
"""Float64 CPU reference of LFM3-VL's vision tower and projector, from the HF processor's pixel_values.

vision_ref64.py MODEL_DIR IMAGE OUT.pt [--hf]   features [tokens, 2048] float64 (+ the tower's output) into OUT.pt;
--hf also runs transformers' Lfm3VlVisionModel + projector in float32 on the same weights and prints the gap.
Weights are bf16 as stored; MLX keeps the depthwise conv as [C, 3, 3, 1], torch as [C, 1, 3, 3].
"""
import importlib.util, json, sys
import torch
from PIL import Image
from safetensors import safe_open

F64 = torch.float64
d, image_path, out_path = sys.argv[1:4]
hf = "--hf" in sys.argv
import os
CONTROL = os.environ.get("LFM3VL_CONTROL", "")  # planted faults the --hf cross-check must see


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


root = json.load(open(f"{d}/config.json"))
vc = root["vision_config"]
hf_loading = False
wmap = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
files = {}


def raw(k):
    f = wmap[k]
    if f not in files:
        files[f] = safe_open(f"{d}/{f}", "pt")
    return files[f].get_tensor(k)


def W(k):
    return raw(k).to(F64)


def conv_weight(k):
    w = raw(k)
    w = w.permute(0, 3, 1, 2).contiguous() if w.shape[-1] == 1 else w  # MLX [C,3,3,1] -> torch [C,1,3,3]
    return w.transpose(2, 3) if CONTROL == "conv_t" and not hf_loading else w


ipm = load_module("lfm3_ip", f"{d}/image_processing_lfm3_vl.py")
icfg = {k: v for k, v in json.load(open(f"{d}/processor_config.json"))["image_processor"].items() if k != "image_processor_type"}
proc = ipm.Lfm3VlImageProcessor(**icfg)
batch = proc(images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
pv, grid = batch["pixel_values"], batch["image_grid_thw"]
t, gh, gw = grid[0].tolist()
merge, H, heads, kvh = vc["spatial_merge_size"], vc["hidden_size"], vc["num_attention_heads"], vc["num_key_value_heads"]
D = H // heads
eps = vc["norm_eps"]


def rms(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def positions():
    hp, wp = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    shape = (gh // merge, merge, gw // merge, merge)
    return hp.reshape(shape).transpose(1, 2).flatten(), wp.reshape(shape).transpose(1, 2).flatten()


def rope_tables():
    spatial = D // 2
    inv = 1.0 / (vc["rope_parameters"]["rope_theta"] ** (torch.arange(0, spatial, 2, dtype=F64) / spatial))
    hp, wp = positions()
    if CONTROL == "rope_hw":
        hp, wp = wp, hp
    fh, fw = hp[:, None].to(F64) * inv, wp[:, None].to(F64) * inv
    f = torch.cat([fh, fw, fh, fw], -1)
    return f.cos(), f.sin()


def rot(x, cos, sin):
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return x * cos[:, None] + torch.cat([-x2, x1], -1) * sin[:, None]


def grid_of(x):  # merge-block order [N, C] -> raster [C, gh, gw]
    if CONTROL == "raster":
        return x.T.reshape(1, -1, gh, gw)
    return x.view(gh // merge, gw // merge, merge, merge, -1).permute(4, 0, 2, 1, 3).reshape(1, -1, gh, gw)


def rows_of(g):  # raster [1, C, gh, gw] -> merge-block order [N, C]
    if CONTROL == "raster":
        return g.reshape(-1, gh * gw).T
    return g.view(-1, gh // merge, merge, gw // merge, merge).permute(1, 3, 2, 4, 0).reshape(gh * gw, -1)


def tower(x):
    pre = "vision_tower."
    h = x @ W(pre + "patchifier.weight").T
    h = rms(h, W(pre + "embedding_norm.weight"))
    cos, sin = rope_tables()
    for i, kind in enumerate(vc["layer_types"]):
        p = f"{pre}layers.{i}."
        a = rms(h, W(p + "operator_norm.weight"))
        if kind == "conv":
            B, C, xx = (a @ W(p + "conv.in_proj.weight").T).chunk(3, -1)
            g = grid_of(B * xx)
            g = torch.nn.functional.conv2d(g, conv_weight(p + "conv.spatial_conv.weight").to(F64), padding=1, groups=H)
            o = (C * rows_of(g)) @ W(p + "conv.out_proj.weight").T
        else:
            n = a.shape[0]
            q = rms((a @ W(p + "self_attn.q_proj.weight").T).view(n, heads, D), W(p + "self_attn.q_layernorm.weight"))
            k = rms((a @ W(p + "self_attn.k_proj.weight").T).view(n, kvh, D), W(p + "self_attn.k_layernorm.weight"))
            v = (a @ W(p + "self_attn.v_proj.weight").T).view(n, kvh, D)
            q, k = rot(q, cos, sin), rot(k, cos, sin)
            k = k.repeat_interleave(heads // kvh, 1)
            v = v.repeat_interleave(heads // kvh, 1)
            s = torch.einsum("qhd,khd->hqk", q, k) * D**-0.5
            o = torch.einsum("hqk,khd->qhd", s.softmax(-1), v).reshape(n, H) @ W(p + "self_attn.out_proj.weight").T
        h = h + o
        f = rms(h, W(p + "ffn_norm.weight"))
        h = h + (torch.nn.functional.silu(f @ W(p + "feed_forward.w1.weight").T) * (f @ W(p + "feed_forward.w3.weight").T)) @ W(p + "feed_forward.w2.weight").T
    return h


def projector(h):
    p = "multi_modal_projector."
    x = torch.nn.functional.layer_norm(h, (H,), W(p + "layer_norm.weight"), W(p + "layer_norm.bias"), root["projector_layer_norm_eps"])
    x = x.reshape(-1, H * root["downsample_factor"] ** 2)
    x = torch.nn.functional.gelu(x @ W(p + "linear_1.weight").T + W(p + "linear_1.bias"))
    return x @ W(p + "linear_2.weight").T + W(p + "linear_2.bias")


with torch.no_grad():
    h = tower(pv.to(F64))
    feats = projector(h)
torch.save({"pixel_values": pv, "grid": grid, "tower": h, "features": feats}, out_path)
print(f"grid {t}x{gh}x{gw}: {gh * gw} patches -> {feats.shape[0]} tokens of {feats.shape[1]}")

if hf:
    import re, tempfile
    hf_loading = True
    tmp = tempfile.mkdtemp()  # the checkpoint's modules with their package-relative imports made absolute
    for fn in os.listdir(d):
        if fn.endswith(".py"):
            open(f"{tmp}/{fn}", "w").write(re.sub(r"from \.(\w+) import", r"from \1 import", open(f"{d}/{fn}").read()))
    sys.path.insert(0, tmp)
    import configuration_lfm3_vl as vm
    import modeling_lfm3_vl as ml
    cfg = vm.Lfm3VlConfig(**{k: v for k, v in root.items() if k not in ("quantization", "quantization_config")})
    cfg.vision_config._attn_implementation = "eager"
    dt = torch.bfloat16 if os.environ.get("LFM3VL_HF_DTYPE") == "bf16" else torch.float32
    vt = ml.Lfm3VlVisionModel(cfg.vision_config).to(dt).eval()
    pj = ml.Lfm3VlMultiModalProjector(cfg).to(dt).eval()
    sd = {k[len("vision_tower."):]: (conv_weight(k) if k.endswith("spatial_conv.weight") else raw(k)).to(dt) for k in wmap if k.startswith("vision_tower.")}
    print("tower load:", vt.load_state_dict(sd, strict=True))
    sd = {k[len("multi_modal_projector."):]: raw(k).to(dt) for k in wmap if k.startswith("multi_modal_projector.")}
    print("projector load:", pj.load_state_dict(sd, strict=True))
    with torch.no_grad():
        ht = vt(pixel_values=pv.to(dt), grid_thw=grid).last_hidden_state
        ft = pj(ht)
    for name, a, b in (("tower", ht, h), ("features", ft, feats)):
        diff = (a.to(F64) - b).abs()
        print(f"{name}: max abs {diff.max().item():.3e}, rel {(diff.norm() / b.norm()).item():.3e}, cos {torch.nn.functional.cosine_similarity(a.to(F64).flatten(), b.flatten(), 0).item():.6f}")
    rows = torch.nn.functional.cosine_similarity(ft.to(F64), feats, 1)
    print(f"features per-row cos: min {rows.min().item():.6f} mean {rows.mean().item():.6f}")
    torch.save({"pixel_values": pv, "grid": grid, "tower": h, "features": feats, "hf_features": ft}, out_path)
