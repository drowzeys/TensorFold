"""Drama 3 prompt rows in the checkpoint's reference-bracket-v1 layout.

Rows are [token id, code 0..9]: text rows carry zero codes, reference audio rows carry
[semantic_begin + code 0, code 0, ..., code 9]. Each literal and caller part is tokenized on its own,
as the reference builder does, since tokenizing the joined string changes ids at the seams.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = "convert the provided text to speech reference to the following:"
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
SEMANTIC_BEGIN_TOKEN = "<|semantic:0|>"
STRUCTURAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:")
MAX_SPEAKER_ID = 15
SPEAKER_IDS = frozenset(str(i) for i in range(MAX_SPEAKER_ID + 1))
SPEAKER_TAG = re.compile(r"<\|speaker:([0-9]+)\|>")
# the Unicode White_Space property (str.strip() alone also strips U+001C..U+001F)
WHITE_SPACE = "".join(map(chr, (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
                          + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000)))


@dataclass
class Reference:
    """A voice reference: its transcript and codes [10, frames]."""

    transcript: str
    codes: np.ndarray


def check_text(text: str, where: str) -> None:
    if not text.strip(WHITE_SPACE):
        raise ValueError(f"{where}: text must not be empty")
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"{where}: {token} is written by the prompt builder, not caller text")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG.match(text, marker.start())
        if tag is None or tag.group(1) not in SPEAKER_IDS:
            raise ValueError(f"{where}: speaker tags are <|speaker:N|> with N in 0..{MAX_SPEAKER_ID}; got {text!r}")


def with_speaker(text: str, speaker: int = 0) -> str:
    """Tag text with ``speaker`` unless a speaker tag leads it after white space."""

    return text if SPEAKER_TAG.match(text.lstrip(WHITE_SPACE)) else f"<|speaker:{speaker}|> {text}"


def build_rows(text: str, references: list[Reference], tokenizer, *, codebooks: int = 10,
               codebook_size: int = 4096) -> np.ndarray:
    """int64 rows [N, 1 + codebooks] ending in the voice turn."""

    semantic_begin = tokenizer.token_to_id(SEMANTIC_BEGIN_TOKEN)
    if semantic_begin is None:
        raise ValueError(f"the tokenizer has no {SEMANTIC_BEGIN_TOKEN} token")
    segments: list = ["<|im_start|>system\n", REFERENCE_INSTRUCTION if references else TTS_INSTRUCTION]
    for i, ref in enumerate(references):
        check_text(ref.transcript, f"reference {i + 1}")
        codes = np.asarray(ref.codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != codebooks or codes.shape[1] == 0:
            raise ValueError(f"reference {i + 1}: codes must be [{codebooks}, frames]; got {codes.shape}")
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise ValueError(f"reference {i + 1}: codes must be in [0, {codebook_size})")
        tagged = with_speaker(ref.transcript, 0)
        body = tagged.lstrip(WHITE_SPACE)
        lead = SPEAKER_TAG.match(body)
        transcript = lead.group(0) + tagged[:len(tagged) - len(body)] + body[lead.end():]
        segments += [f"\n\n[Reference {i + 1}]\nText: {transcript}\nSpeech: ", codes]
    check_text(text, "input")
    segments += [MESSAGE_END, "<|im_start|>user\n", with_speaker(text, 0), MESSAGE_END, GENERATION_START]
    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = np.asarray(tokenizer.encode(segment).ids, dtype=np.int64)
            block = np.zeros((ids.size, codebooks + 1), dtype=np.int64)
            block[:, 0] = ids
        else:
            block = np.concatenate([segment[:1] + semantic_begin, segment], axis=0).T
        rows.append(block)
    return np.concatenate(rows, axis=0)
