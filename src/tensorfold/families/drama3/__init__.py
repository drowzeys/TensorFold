"""Fish Audio Drama 3 text to speech (MLX): Qwen3.5 MoE backbone, codebook tower and Fish codec.

Built with Fish Audio. The checkpoint is under the Fish Audio Community License v1.0; TensorFold
never ships or redistributes its weights. See DRAMA3.md at the repository root.
"""

from __future__ import annotations

# Not a lane family: one forward samples a whole audio frame (a main token and nine codes), so the
# text engine's lanes and drafting do not apply. Entry point: `python -m tensorfold.drama3`.
MODEL_TYPES = ("fish_omni",)
TITLE = "Fish Audio Drama 3"
LANES = False
MODELS = ()
CONVERTED_FORMAT = "tensorfold-drama3-mlx-v1"


def is_checkpoint(model_dir) -> bool:
    """Whether ``model_dir`` holds a Drama 3 checkpoint (Fish's own or the converted MLX one)."""

    import json
    from pathlib import Path

    config = Path(model_dir) / "config.json"
    if not config.is_file():
        return False
    return json.loads(config.read_text()).get("model_type") in MODEL_TYPES
