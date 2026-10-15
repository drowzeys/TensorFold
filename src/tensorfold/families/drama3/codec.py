"""The Drama 3 codec: frames of 10 codes to 44.1 kHz audio, and reference audio to codes (voice cloning)."""

from __future__ import annotations

import math
from pathlib import Path

SAMPLE_RATE = 44100
CODEBOOKS = 10
# codec.pth buffers the MLX model rebuilds itself
SKIPPED = ("causal_mask", "freqs_cis")


def convert_codec(codec_pth: Path) -> dict:
    """codec.pth (PyTorch) -> float32 MLX tensors under the MLX codec's names. Needs torch."""

    import mlx.core as mx
    import torch

    state = torch.load(str(codec_pth), map_location="cpu", mmap=True, weights_only=True)
    return {name: mx.array(value.float().numpy()) for name, value in state.items()
            if not name.endswith(SKIPPED) and value.dtype != torch.bool}


class Codec:
    """The MLX Fish codec with weights loaded and checked (no tensor may be left at its random start)."""

    def __init__(self, path: Path, dtype=None) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from tensorfold.families.drama3.codec_dac import build_ae

        self.model = build_ae()
        weights = self.model.sanitize(mx.load(str(path)))
        # the inner conv of a weight-normed conv is set from weight_g / weight_v; masks and rotary tables are built
        derived = (".conv.conv.weight", ".conv.conv.bias", "causal_mask", "freqs_cis")
        expected = {name for name, _ in tree_flatten(self.model.parameters()) if not name.endswith(derived)}
        missing = expected - set(weights)
        unknown = set(weights) - expected
        if missing or unknown:
            raise ValueError(f"codec weights do not match the model: missing {sorted(missing)[:6]}, "
                             f"unknown {sorted(unknown)[:6]}")
        if dtype is not None:
            weights = {k: v.astype(dtype) for k, v in weights.items()}
        self.model.load_weights(list(weights.items()), strict=False)
        self.model.eval()
        for _, module in self.model.named_modules():
            if hasattr(module, "fold"):
                module.fold()
        mx.eval(self.model.parameters())
        self.sample_rate = SAMPLE_RATE
        self.frame_length = int(self.model.frame_length)       # samples a frame (2048)

    def decode(self, codes, window: int = 2048, context: int = 64):
        """codes [10, frames] (int) -> float32 waveform [frames * 2048].

        The quantizer's decode (codebook sums, the windowed transformer and upsampling) runs over the whole clip;
        the convolutional decoder, which holds the large activations, runs in windows of ``window`` latent steps
        (4 a frame) with ``context`` steps of left context whose samples are dropped (it is causal, and 32 steps
        already reproduce the one-pass output to float noise).
        """

        import mlx.core as mx

        codes = mx.array(codes).astype(mx.int32)
        codes = mx.concatenate([mx.clip(codes[:1], 0, 4095), mx.clip(codes[1:], 0, 1023)], axis=0)
        if int(codes.shape[1]) == 0:
            return mx.zeros((0,), dtype=mx.float32)
        latent = self.model.quantizer.decode(codes[None])
        mx.eval(latent)
        steps = int(latent.shape[-1])
        parts = []
        for start in range(0, steps, window):
            first = max(0, start - context)
            audio = self.model.decoder(latent[:, :, first:min(steps, start + window)]).reshape(-1)
            audio = audio[(start - first) * (int(audio.shape[0]) // (min(steps, start + window) - first)):]
            mx.eval(audio)
            parts.append(audio)
        return mx.concatenate(parts).astype(mx.float32)

    def encode(self, waveform):
        """float32 mono waveform at 44.1 kHz [samples] -> codes [10, frames] (int32)."""

        import mlx.core as mx

        audio = mx.array(waveform, dtype=mx.float32).reshape(1, 1, -1)
        codes, lengths = self.model.encode(audio)
        frames = int(math.ceil(int(audio.shape[-1]) / self.frame_length))
        return codes[0, :, :frames].astype(mx.int32)


def load_audio(path: str | Path, sample_rate: int = SAMPLE_RATE):
    """A mono float32 numpy waveform at ``sample_rate`` (soundfile reads it; numpy polyphase-free resampling)."""

    import numpy as np
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        audio = resample(audio, rate, sample_rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


def resample(audio, rate: int, target: int):
    """Band-limited resampling with a windowed-sinc kernel (torchaudio's default 'sinc_interp_hann', width 6)."""

    import numpy as np

    g = math.gcd(rate, target)
    orig, new = rate // g, target // g
    width = 6
    rolloff = 0.99
    base = min(orig, new) * rolloff
    span = int(math.ceil(width * orig / base))
    idx = np.arange(-span, span + orig, dtype=np.float64)[None, :] / orig
    t = (-np.arange(new, dtype=np.float64)[:, None] / new + idx) * base
    t = np.clip(t, -width, width)
    window = np.cos(t * math.pi / width / 2) ** 2
    t = t * math.pi
    kernels = np.where(t == 0, 1.0, np.sin(t) / np.where(t == 0, 1.0, t)) * window * base / orig
    length = audio.shape[-1]
    padded = np.pad(audio.astype(np.float64), (span, span + orig))
    frames = (length + orig - 1) // orig + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernels.shape[1])[::orig][:frames]
    out = (windows @ kernels.T).reshape(-1)
    return out[: int(math.ceil(new * length / orig))].astype(np.float32)


def save_wav(path: str | Path, audio, sample_rate: int = SAMPLE_RATE) -> None:
    import numpy as np
    import soundfile

    soundfile.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")
