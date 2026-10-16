"""MiniMax Music 3 prompts: caption and lyrics to the token ids the AR model reads.

The layout is ``<|im_start|><|caption_start|>caption<|caption_end|><|lyrics_start|>[start]\\nlyrics<|lyrics_end|>
<|im_end|><|audio_start|>``. Captions lose markdown and ``<|tag value|>`` markers (read as "tag is value");
lyric section tags such as ``[Verse]`` go on lines of their own, in lower case. The unconditional prompt for
classifier-free guidance keeps the first token and the last two and turns everything between into ``<|audio_cfg|>``.
"""

from __future__ import annotations

import re

SPECIAL = {
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|audio_cfg|>": 151654,
    "<|audio_start|>": 151669,
    "<|audio_end|>": 151670,
    "<|caption_start|>": 151671,
    "<|caption_end|>": 151672,
    "<|lyrics_start|>": 151673,
    "<|lyrics_end|>": 151674,
}
AUDIO_CODE_OFFSET = 151675            # first audio-code id of the full vocabulary (the pruned files drop the rest)
MAX_PROMPT_TOKENS = 5000

_TAG = re.compile(r"<\|([^|]*)\|>")
_SECTION = re.compile(r"\s*(\[[^\]]+\])\s*")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BULLET = re.compile(r"^\s*[*+-]\s+")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_RULE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)


def _tag_words(match: re.Match) -> str:
    words = match.group(1).strip().split(None, 1)
    return f"{words[0]} is {words[1]}" if len(words) == 2 else match.group(1).strip()


def _plain_line(line: str) -> str:
    line = _BULLET.sub("", _HEADING.sub("", line))
    while "**" in line:
        unbolded = _BOLD.sub(r"\1", line)
        if unbolded == line:
            break
        line = unbolded
    return _ITALIC.sub(r"\1", line).rstrip()


def clean_caption(caption: str) -> str:
    text = "\n".join(_plain_line(line) for line in _TAG.sub(_tag_words, caption).splitlines())
    text = _RULE.sub("", text).replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    pieces = [p.lower() if p.startswith("[") else p for p in _SECTION.split(lyrics) if p]
    return "[start]\n" + "\n".join(pieces).replace(" ^ ", "\n")


def prompt_text(caption: str, lyrics: str) -> str:
    return ("<|im_start|><|caption_start|>" + clean_caption(caption) + "<|caption_end|><|lyrics_start|>"
            + normalize_lyrics(lyrics) + "<|lyrics_end|><|im_end|><|audio_start|>")


def check_tokenizer(tokenizer) -> None:
    for token, expected in SPECIAL.items():
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"MiniMax Music 3 tokenizer: {token} is {tokenizer.token_to_id(token)}, not {expected}")


def encode(tokenizer, caption: str, lyrics: str) -> list[int]:
    ids = tokenizer.encode(prompt_text(caption, lyrics), add_special_tokens=False).ids
    if len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"the prompt has {len(ids)} tokens; MiniMax Music 3 takes at most {MAX_PROMPT_TOKENS}")
    return ids


def unconditional(ids: list[int]) -> list[int]:
    return [ids[0]] + [SPECIAL["<|audio_cfg|>"]] * (len(ids) - 3) + ids[-2:]
