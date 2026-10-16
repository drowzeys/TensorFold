"""ComfyUI reference for MiniMax Music 3 (run from ~/ComfyUI with its venv; torch MPS).

    ref_comfy.py ar   --out DIR --caption-file C --lyrics-file L --seconds S --seed N [--ar-cfg 1.5]
    ref_comfy.py dit  --out DIR --seed N [--steps 30 --dit-cfg 1.7]     (needs DIR/hidden.npy from `ar`)
    ref_comfy.py dav  --out DIR                                          (needs DIR/latent.npy from `dit`)
    ref_comfy.py probe --out DIR                                         (fixed-input DiT and DAV outputs)

Each stage runs in its own process so `/usr/bin/time -l` gives its peak footprint. Uses ComfyUI's own loaders,
nodes and sampler; hooks only record tensors.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

import comfy.model_management as mm
import comfy.sd
import comfy.utils
import comfy.sample
import nodes
from comfy_extras.nodes_audio import vae_decode_audio
from comfy.ldm.minimax_music.ar import AUDIO_FRAMES_PER_SECOND, MAX_AUDIO_FRAMES

M = os.path.expanduser("~/studio/models/MiniMax-Music3-Comfy")
TE = f"{M}/text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors"
DIT = f"{M}/diffusion_models/minimax_music3_dit_fp16.safetensors"
DAV = f"{M}/vae/minimax_music3_dav.safetensors"


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def stage_ar(a):
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip([TE], clip_type=comfy.sd.CLIPType.MINIMAX)
    te = clip.cond_stage_model
    load_s = time.perf_counter() - t0
    rec = {"c0": [], "depth": [], "codes": [], "token": []}
    te.model.lm_head_pruned.register_forward_hook(lambda m, i, o: rec["c0"].append(o.detach().float().cpu().half()))
    for head in te.model.audio_decoder.audio_heads:
        head.register_forward_hook(lambda m, i, o: rec["depth"].append(o.detach().float().cpu()))
    orig_depth, orig_c0 = te._depth_codes, te._sample_c0

    def depth_codes(*args, **kw):
        codes, hidden = orig_depth(*args, **kw)
        rec["codes"].append(codes[0].detach().cpu())
        return codes, hidden

    def sample_c0(*args, **kw):
        out = orig_c0(*args, **kw)
        rec["token"].append(out[1].detach().cpu())
        return out
    te._depth_codes, te._sample_c0 = depth_codes, sample_c0
    caption, lyrics = open(a.caption_file).read(), open(a.lyrics_file).read()
    frames = min(MAX_AUDIO_FRAMES, max(1, round(a.seconds * AUDIO_FRAMES_PER_SECOND)))
    t1 = time.perf_counter()
    tokens = clip.tokenize(caption, lyrics=lyrics, seed=a.seed, max_audio_frames=frames, cfg_scale=a.ar_cfg, top_k=50)
    cond = clip.encode_from_tokens_scheduled(tokens)
    sync()
    ar_s = time.perf_counter() - t1
    hidden = cond[0][0][0].float().cpu().numpy()                        # [T, 32768]
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/hidden.npy", hidden.astype(np.float32))
    np.save(f"{a.out}/codes.npy", torch.stack(rec["codes"]).numpy().astype(np.int32))
    np.save(f"{a.out}/tokens.npy", torch.cat(rec["token"]).numpy().astype(np.int64))
    if a.save_logits:
        np.save(f"{a.out}/c0_logits.npy", torch.stack(rec["c0"]).numpy())
        d = torch.stack(rec["depth"])                                        # [F*7, 2, 1024]
        np.save(f"{a.out}/depth_logits.npy", d.reshape(-1, 7, *d.shape[1:]).numpy())
    ids = tokens["minimax_music3"][0]
    stats = {"stage": "ar", "load_s": load_s, "ar_s": ar_s, "frames": int(hidden.shape[0]),
             "audio_s": hidden.shape[0] / 25, "prompt_tokens": len(ids), "frames_per_s": hidden.shape[0] / ar_s,
             "dtype": str(next(te.parameters()).dtype), "device": str(mm.get_torch_device())}
    json.dump({"ids": [int(t) for t, _ in ids]}, open(f"{a.out}/ids.json", "w"))
    return stats


def load_dit():
    model = comfy.sd.load_diffusion_model(DIT)
    return model


def stage_dit(a):
    t0 = time.perf_counter()
    model = load_dit()
    load_s = time.perf_counter() - t0
    hidden = torch.from_numpy(np.load(f"{a.out}/hidden.npy")).to(torch.bfloat16)[None]
    pos = [[hidden, {"conditioning_scale": torch.ones((1, 1, 1), dtype=hidden.dtype)}]]
    neg = nodes.ConditioningZeroOut().zero_out(pos)[0]
    from comfy.ldm.minimax_music.dit import latent_length
    latent = {"samples": torch.zeros((1, 128, latent_length(hidden.shape[1]))), "type": "audio",
              "downscale_ratio_temporal": 512}
    t1 = time.perf_counter()
    out = nodes.common_ksampler(model, a.seed, a.steps, a.dit_cfg, "euler", "simple", pos, neg, latent)[0]
    sync()
    dit_s = time.perf_counter() - t1
    np.save(f"{a.out}/latent.npy", out["samples"].float().cpu().numpy())
    noise = comfy.sample.prepare_noise(latent["samples"], a.seed)
    np.save(f"{a.out}/noise.npy", noise.float().numpy())
    return {"stage": "dit", "load_s": load_s, "dit_s": dit_s, "latent_frames": int(latent["samples"].shape[-1]),
            "dtype": str(model.model.get_dtype()), "steps": a.steps}


def stage_dav(a):
    t0 = time.perf_counter()
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(DAV))
    load_s = time.perf_counter() - t0
    latent = torch.from_numpy(np.load(f"{a.out}/latent.npy"))
    t1 = time.perf_counter()
    audio = vae_decode_audio(vae, {"samples": latent})
    sync()
    dav_s = time.perf_counter() - t1
    wave = audio["waveform"][0].float().cpu().numpy()
    np.save(f"{a.out}/audio.npy", wave)
    import soundfile as sf
    sf.write(f"{a.out}/song.wav", wave.T, audio["sample_rate"], subtype="PCM_16")
    return {"stage": "dav", "load_s": load_s, "dav_s": dav_s, "audio_s": wave.shape[1] / audio["sample_rate"],
            "sample_rate": audio["sample_rate"]}


def stage_probe(a):
    """Fixed inputs: DiT model output at three sigmas (cond + zeroed cond) on probe_hidden.npy / probe_x.npy, and
    the DAV decode of probe_latent.npy."""
    model = load_dit()
    mm.load_models_gpu([model])
    dm = model.model.diffusion_model
    dev, dt = mm.get_torch_device(), model.model.get_dtype()
    hidden = torch.from_numpy(np.load(f"{a.out}/probe_hidden.npy")).to(dev, dt)[None]
    x = torch.from_numpy(np.load(f"{a.out}/probe_x.npy")).to(dev, dt)[None]
    outs = {}
    for sigma in (1.0, 0.5, 0.1):
        t = torch.full((1,), 1.0 - sigma, device=dev, dtype=torch.float32)
        scale = torch.ones((1, 1, 1), device=dev, dtype=dt)
        with torch.no_grad():
            c = dm(x, t, hidden, scale).float().cpu().numpy()[0]
            u = dm(x, t, torch.zeros_like(hidden), scale).float().cpu().numpy()[0]
        outs[f"cond_{sigma}"], outs[f"uncond_{sigma}"] = c, u
    np.savez(f"{a.out}/probe_dit.npz", **outs)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(DAV))
    lat = torch.from_numpy(np.load(f"{a.out}/probe_latent.npy"))[None]
    with torch.no_grad():
        audio = vae.decode(lat).movedim(-1, 1)[0].float().cpu().numpy()
    np.save(f"{a.out}/probe_dav.npy", audio)
    return {"stage": "probe", "dit_dtype": str(dt), "dit_window_frames": int(x.shape[-1])}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("ar", "dit", "dav", "probe"))
    p.add_argument("--out", required=True)
    p.add_argument("--caption-file")
    p.add_argument("--lyrics-file")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ar-cfg", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--dit-cfg", type=float, default=1.7)
    p.add_argument("--save-logits", action="store_true")
    a = p.parse_args()
    a.out = os.path.expanduser(a.out)
    os.makedirs(a.out, exist_ok=True)
    with torch.inference_mode():
        stats = globals()[f"stage_{a.stage}"](a)
    print(json.dumps(stats))
    with open(f"{a.out}/stats_{a.stage}.json", "w") as f:
        json.dump(stats, f, indent=1)
