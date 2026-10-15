#!/usr/bin/env python3
"""Phase-1 codec for drama3-generate: frames (int32 [n, 11], main token + codes 0..9) -> 44.1 kHz WAV through the Fish
S2 modded DAC (the oracle's fishcodec copy, ~/drama3/oracle), and voice-reference encoding (WAV -> int32 [frames, 10]
codes for --ref-codes).
usage: frames_to_wav.py decode FRAMES WAV [FRAMES WAV ...]
       frames_to_wav.py encode REF.wav CODES"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.environ.get("ORACLE_DIR", "/drama3/oracle"))
from drama3_ref import build_codec, decode_frames, encode_reference, write_wav  # noqa: E402

MODEL = os.environ.get("DRAMA3_MODEL", "/w")


def main() -> None:
    dev = torch.device("cuda")
    codec = build_codec(MODEL, dev)
    if sys.argv[1] == "decode":
        for src, dst in zip(sys.argv[2::2], sys.argv[3::2]):
            f = torch.from_numpy(np.fromfile(src, dtype=np.int32).reshape(-1, 11).astype(np.int64))
            audio = decode_frames(codec, f, dev)
            write_wav(dst, audio)
            print(f"{src}: {f.shape[0]} frames -> {dst} ({audio.numel() / 44100:.2f} s)")
    elif sys.argv[1] == "encode":
        codes, _ = encode_reference(codec, sys.argv[2], dev)
        codes.T.contiguous().numpy().astype(np.int32).tofile(sys.argv[3])
        print(f"{sys.argv[2]}: {codes.shape[1]} frames -> {sys.argv[3]}")


if __name__ == "__main__":
    main()
