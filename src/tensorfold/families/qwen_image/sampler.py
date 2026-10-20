"""Qwen-Image-2.1 denoising loop: Euler over the shifted schedule, no guidance."""

from __future__ import annotations

from .config import latent_grid
from .schedule import sigmas as make_sigmas


def start_noise(seed: int, width: int, height: int, channels: int = 64):
    """The starting latents, (1, rows * columns, channels) bfloat16; the draw mflux makes for the same seed."""

    import mlx.core as mx

    rows, columns = latent_grid(width, height)
    return mx.random.normal(shape=[1, rows * columns, channels], key=mx.random.key(seed)).astype(mx.bfloat16)


def torch_noise(seed: int, width: int, height: int, channels: int = 64):
    """The starting latents (1, rows * columns, channels) float32 that ComfyUI and diffusers draw for ``seed`` on the
    CPU: torch.randn((1, channels, rows, columns), generator=torch.manual_seed(seed)), reproduced in numpy to 2e-6.
    With it a seed gives the same picture here as on a ComfyUI or diffusers install (``start_noise`` gives a
    different, equally valid picture for the same seed).

    torch seeds an MT19937 with the seed's low 32 bits and fills a float tensor 16 values at a time: 16 uniforms from
    24-bit draws, then Box-Muller on the pairs (j, j + 8).
    """

    import mlx.core as mx
    import numpy as np

    rows, columns = latent_grid(width, height)
    count = channels * rows * columns
    if count % 16:
        raise ValueError("the noise needs a multiple of 16 values")
    bits = np.random.MT19937(0)
    bits._legacy_seeding(seed & 0xFFFFFFFF)
    uniform = ((bits.random_raw(count) & ((1 << 24) - 1)).astype(np.float64) * 2.0**-24).astype(np.float32)
    block = uniform.reshape(-1, 16)
    radius = np.sqrt(np.float32(-2.0) * np.log(np.float32(1.0) - block[:, :8]))
    theta = np.float32(2.0 * np.pi) * block[:, 8:]
    noise = np.concatenate([radius * np.cos(theta), radius * np.sin(theta)], axis=1).reshape(channels, rows, columns)
    return mx.array(noise.transpose(1, 2, 0).reshape(1, rows * columns, channels))


def denoise(dit, text, width: int, height: int, steps: int = 40, seed: int = 0, nodes=None, on_step=None,
            latents=None, shift_nodes: bool = True):
    """Image latents (1, channels, rows, columns) float32 from prompt embeddings ``text`` (1, tokens, dim).

    ``nodes`` are fixed noise levels; ``steps`` is ignored when they are given. ``shift_nodes`` as in
    ``schedule.sigmas``: True for an adapter's training nodes, False for a checkpoint's own saved schedule.
    ``on_step(index, total)`` is called after each step.
    """

    import mlx.core as mx

    rows, columns = latent_grid(width, height)
    levels = make_sigmas(steps, width, height, nodes, shift_nodes)
    prefix = dit.prefix(text, rows, columns)
    x = start_noise(seed, width, height, dit.config.in_channels) if latents is None else latents
    x = x.astype(mx.float32)
    total = len(levels) - 1
    for index in range(total):
        velocity = dit(x, float(levels[index]), prefix)
        x = x + velocity.astype(mx.float32) * float(levels[index + 1] - levels[index])
        mx.eval(x)
        if on_step is not None:
            on_step(index, total)
    return x.reshape(1, rows, columns, -1).transpose(0, 3, 1, 2)
