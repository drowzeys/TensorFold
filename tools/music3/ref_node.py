"""MiniMax Music 3 reference dumps from inside the Studio's ComfyUI server (a custom node; never a standalone ComfyUI).

Put this file's folder on the server's custom_nodes path (an --extra-model-paths-config with `custom_nodes:`), start
the server with the Studio's spark/comfy.sh, then drive it with `ref_node.py submit STAGE --out DIR ...` (the client
half of this file, plain urllib). Stages, each a prompt of one TFMusic3RefDump node:

  ids    prompt ids of the cases in --cases (a JSON list of [caption, lyrics]) -> DIR/ids_cases.json
  ar     ComfyUI's AR on caption/lyrics -> hidden.npy [T, 32768], codes.npy [F, 8], tokens.npy, ids.json and with
         --save-logits c0_logits.npy [F, 2, 16385] (fp16) and depth_logits.npy [F, 7, 2, 1024]
  dit    KSampler (euler, simple) over DIR/hidden.npy with the zeroed condition as negative -> latent.npy, noise.npy
  dav    VAE decode of DIR/latent.npy (with the level rule) -> audio.npy [2, N], song.wav
  probe  fixed inputs (the first 200 frame states, seeded noise, the latent's first 689 rows): the DiT's output at
         sigma 1.0 / 0.5 / 0.1 for both branches -> probe_dit.npz; the DAV of probe_latent.npy -> probe_dav.npy

Hooks only record tensors; the loaders, nodes and sampler are ComfyUI's own (the MLX port's tools/music3_ref_comfy.py,
moved into the server). Each stage frees its models when it ends.
"""
import json
import os
import sys
import time

STAGES = ("ids", "ar", "dit", "dav", "probe")


def _paths():
    import folder_paths
    return (folder_paths.get_full_path_or_raise("text_encoders", "minimax_music3_text_encoder_pruned_bf16.safetensors"),
            folder_paths.get_full_path_or_raise("diffusion_models", "minimax_music3_dit_fp16.safetensors"),
            folder_paths.get_full_path_or_raise("vae", "minimax_music3_dav.safetensors"))


def _free():
    import gc
    import torch
    import comfy.model_management as mm
    mm.unload_all_models()
    mm.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def stage_ids(a):
    import comfy.sd
    from comfy.ldm.minimax_music.prompt import build_prompt
    te, _, _ = _paths()
    clip = comfy.sd.load_clip([te], clip_type=comfy.sd.CLIPType.MINIMAX)
    tok = clip.tokenizer.minimax_music3 if hasattr(clip.tokenizer, "minimax_music3") else clip.tokenizer
    inner = getattr(tok, "tokenizer", None)
    out = []
    for caption, lyrics in json.load(open(a["cases"])):
        ids = clip.tokenize(caption, lyrics=lyrics)["minimax_music3"][0]
        out.append({"caption": caption, "lyrics": lyrics, "ids": [int(t) for t, _ in ids],
                    "prompt": build_prompt(caption, lyrics)})
    json.dump(out, open(os.path.join(a["out"], "ids_cases.json"), "w"), indent=1)
    del clip, inner
    return {"stage": "ids", "cases": len(out)}


def stage_ar(a):
    import numpy as np
    import torch
    import comfy.sd
    import comfy.model_management as mm
    from comfy.ldm.minimax_music.ar import AUDIO_FRAMES_PER_SECOND, MAX_AUDIO_FRAMES
    te, _, _ = _paths()
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip([te], clip_type=comfy.sd.CLIPType.MINIMAX)
    tm = clip.cond_stage_model
    model = getattr(tm, "minimax_music3", tm)
    load_s = time.perf_counter() - t0
    rec = {"c0": [], "depth": [], "codes": [], "token": []}
    hooks = []
    if a.get("save_logits"):
        hooks.append(model.model.lm_head_pruned.register_forward_hook(lambda m, i, o: rec["c0"].append(o.detach().float().cpu().half())))
        for head in model.model.audio_decoder.audio_heads:
            hooks.append(head.register_forward_hook(lambda m, i, o: rec["depth"].append(o.detach().float().cpu())))
    orig_depth, orig_c0 = model._depth_codes, model._sample_c0

    def depth_codes(*args, **kw):
        codes, hidden = orig_depth(*args, **kw)
        rec["codes"].append(codes[0].detach().cpu())
        return codes, hidden

    def sample_c0(*args, **kw):
        out = orig_c0(*args, **kw)
        rec["token"].append(out[1].detach().cpu())
        return out
    model._depth_codes, model._sample_c0 = depth_codes, sample_c0
    caption, lyrics = open(a["caption_file"]).read(), open(a["lyrics_file"]).read()
    frames = min(MAX_AUDIO_FRAMES, max(1, round(float(a["seconds"]) * AUDIO_FRAMES_PER_SECOND)))
    t1 = time.perf_counter()
    tokens = clip.tokenize(caption, lyrics=lyrics, seed=int(a["seed"]), max_audio_frames=frames,
                           cfg_scale=float(a.get("ar_cfg", 1.5)), top_k=50)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ar_s = time.perf_counter() - t1
    for h in hooks:
        h.remove()
    model._depth_codes, model._sample_c0 = orig_depth, orig_c0
    hidden = cond[0][0][0].float().cpu().numpy()
    out = a["out"]
    np.save(f"{out}/hidden.npy", hidden.astype(np.float32))
    np.save(f"{out}/codes.npy", torch.stack(rec["codes"]).numpy().astype(np.int32))
    np.save(f"{out}/tokens.npy", torch.cat(rec["token"]).numpy().astype(np.int64))
    if a.get("save_logits"):
        np.save(f"{out}/c0_logits.npy", torch.stack(rec["c0"]).numpy())
        d = torch.stack(rec["depth"])
        np.save(f"{out}/depth_logits.npy", d.reshape(-1, 7, *d.shape[1:]).numpy())
    ids = tokens["minimax_music3"][0]
    json.dump({"ids": [int(t) for t, _ in ids]}, open(f"{out}/ids.json", "w"))
    stats = {"stage": "ar", "load_s": load_s, "ar_s": ar_s, "frames": int(hidden.shape[0]),
             "audio_s": hidden.shape[0] / 25, "prompt_tokens": len(ids), "frames_per_s": hidden.shape[0] / ar_s,
             "dtype": str(next(model.parameters()).dtype), "device": str(mm.get_torch_device())}
    del clip, model, tm, cond
    return stats


def stage_dit(a):
    import numpy as np
    import torch
    import comfy.sd
    import comfy.sample
    import nodes
    from comfy.ldm.minimax_music.dit import latent_length
    _, dit, _ = _paths()
    t0 = time.perf_counter()
    model = comfy.sd.load_diffusion_model(dit)
    load_s = time.perf_counter() - t0
    out = a["out"]
    hidden = torch.from_numpy(np.load(f"{out}/hidden.npy")).to(torch.bfloat16)[None]
    pos = [[hidden, {"conditioning_scale": torch.ones((1, 1, 1), dtype=hidden.dtype)}]]
    neg = nodes.ConditioningZeroOut().zero_out(pos)[0]
    latent = {"samples": torch.zeros((1, 128, latent_length(hidden.shape[1]))), "type": "audio", "downscale_ratio_temporal": 512}
    t1 = time.perf_counter()
    res = nodes.common_ksampler(model, int(a["seed"]), int(a.get("steps", 30)), float(a.get("dit_cfg", 1.7)), "euler", "simple", pos, neg, latent)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dit_s = time.perf_counter() - t1
    np.save(f"{out}/latent.npy", res["samples"].float().cpu().numpy())
    noise = comfy.sample.prepare_noise(latent["samples"], int(a["seed"]))
    np.save(f"{out}/noise.npy", noise.float().numpy())
    stats = {"stage": "dit", "load_s": load_s, "dit_s": dit_s, "latent_frames": int(latent["samples"].shape[-1]),
             "dtype": str(model.model.get_dtype()), "steps": int(a.get("steps", 30))}
    del model
    return stats


def stage_dav(a):
    import numpy as np
    import torch
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_audio import vae_decode_audio
    _, _, dav = _paths()
    t0 = time.perf_counter()
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(dav))
    load_s = time.perf_counter() - t0
    out = a["out"]
    latent = torch.from_numpy(np.load(f"{out}/latent.npy"))
    t1 = time.perf_counter()
    audio = vae_decode_audio(vae, {"samples": latent})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dav_s = time.perf_counter() - t1
    wave = audio["waveform"][0].float().cpu().numpy()
    np.save(f"{out}/audio.npy", wave)
    stats = {"stage": "dav", "load_s": load_s, "dav_s": dav_s, "audio_s": wave.shape[1] / audio["sample_rate"],
             "sample_rate": audio["sample_rate"], "vae_dtype": str(getattr(vae, "vae_dtype", ""))}
    del vae
    return stats


def stage_probe(a):
    import numpy as np
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_management as mm
    _, dit, dav = _paths()
    out = a["out"]
    hidden_all = np.load(f"{out}/hidden.npy")
    np.save(f"{out}/probe_hidden.npy", hidden_all[:200].astype(np.float32))
    np.save(f"{out}/probe_x.npy", np.random.default_rng(0).standard_normal((128, 689)).astype(np.float32))
    np.save(f"{out}/probe_latent.npy", np.load(f"{out}/latent.npy")[0, :, :689].astype(np.float32))
    model = comfy.sd.load_diffusion_model(dit)
    mm.load_models_gpu([model])
    dm = model.model.diffusion_model
    dev, dt = mm.get_torch_device(), model.model.get_dtype()
    hidden = torch.from_numpy(np.load(f"{out}/probe_hidden.npy")).to(dev, dt)[None]
    x = torch.from_numpy(np.load(f"{out}/probe_x.npy")).to(dev, dt)[None]
    outs = {}
    for sigma in (1.0, 0.5, 0.1):
        t = torch.full((1,), 1.0 - sigma, device=dev, dtype=torch.float32)
        scale = torch.ones((1, 1, 1), device=dev, dtype=dt)
        with torch.no_grad():
            c = dm(x, t, hidden, scale).float().cpu().numpy()[0]
            u = dm(x, t, torch.zeros_like(hidden), scale).float().cpu().numpy()[0]
        outs[f"cond_{sigma}"], outs[f"uncond_{sigma}"] = c, u
    np.savez(f"{out}/probe_dit.npz", **outs)
    del model, dm
    _free()
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(dav))
    lat = torch.from_numpy(np.load(f"{out}/probe_latent.npy"))[None]
    with torch.no_grad():
        audio = vae.decode(lat).movedim(-1, 1)[0].float().cpu().numpy()
    np.save(f"{out}/probe_dav.npy", audio)
    del vae
    return {"stage": "probe", "dit_dtype": str(dt), "dit_window_frames": int(x.shape[-1])}


class TFMusic3RefDump:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"request": ("STRING", {"default": "{}", "multiline": True})}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "tensorfold/test"

    def run(self, request):
        import torch
        a = json.loads(request)
        os.makedirs(a["out"], exist_ok=True)
        try:
            with torch.inference_mode():
                stats = globals()["stage_" + a["stage"]](a)
            json.dump(stats, open(os.path.join(a["out"], f"stats_{a['stage']}.json"), "w"), indent=1)
        finally:
            _free()
        return ()


NODE_CLASS_MAPPINGS = {"TFMusic3RefDump": TFMusic3RefDump}


# ---------------------------------------------------------------------------------------------- the client half

def submit(argv):
    import argparse
    import urllib.request
    p = argparse.ArgumentParser(prog="ref_node.py submit")
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8190")
    p.add_argument("--caption-file")
    p.add_argument("--lyrics-file")
    p.add_argument("--cases")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ar-cfg", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dit-cfg", type=float, default=1.7)
    p.add_argument("--save-logits", action="store_true")
    p.add_argument("--timeout", type=float, default=3600)
    a = p.parse_args(argv)
    req = {"stage": a.stage, "out": os.path.abspath(os.path.expanduser(a.out)), "seconds": a.seconds, "seed": a.seed,
           "ar_cfg": a.ar_cfg, "steps": a.steps, "dit_cfg": a.dit_cfg, "save_logits": a.save_logits, "nonce": time.time()}
    for k in ("caption_file", "lyrics_file", "cases"):
        if getattr(a, k):
            req[k] = os.path.abspath(os.path.expanduser(getattr(a, k)))
    body = json.dumps({"prompt": {"1": {"class_type": "TFMusic3RefDump", "inputs": {"request": json.dumps(req)}}}}).encode()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(a.url + "/prompt", body, {"Content-Type": "application/json"})).read())
    pid = r["prompt_id"]
    began = time.time()
    while time.time() - began < a.timeout:
        time.sleep(2)
        h = json.loads(urllib.request.urlopen(f"{a.url}/history/{pid}").read())
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                print(json.dumps(st, indent=1)[-4000:])
                return 1
            if st.get("completed"):
                f = os.path.join(req["out"], f"stats_{a.stage}.json")
                print(open(f).read() if os.path.exists(f) else "done")
                return 0
    print("timeout")
    return 1


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "submit":
    sys.exit(submit(sys.argv[2:]))
