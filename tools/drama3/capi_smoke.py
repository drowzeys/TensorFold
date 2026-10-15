#!/usr/bin/env python3
"""libtensorfold_drama3.so smoke test (ctypes): open with the codec, encode a reference WAV, build prompt rows
(Zig tokenizer), stream speech through a PCM callback, and the one-call text -> WAV.
usage: capi_smoke.py LIB MODEL_DIR CODEC REF.wav "REF TEXT" "TEXT" OUT_PREFIX"""
import ctypes as C
import sys
import time
import wave

lib = C.CDLL(sys.argv[1])
lib.tf_drama3_open2.restype = C.c_void_p
lib.tf_drama3_open2.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int]
lib.tf_drama3_error.restype = C.c_char_p
lib.tf_drama3_encode_wav.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int32), C.c_int]
lib.tf_drama3_prompt.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_void_p, C.c_int, C.POINTER(C.c_int32), C.c_int]
CB = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(C.c_float), C.c_int)
lib.tf_drama3_speak.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_int, C.c_float, C.c_int, C.c_float, C.c_uint64, C.c_int, C.c_int, C.c_int, CB, C.c_void_p, C.POINTER(C.c_int32), C.POINTER(C.c_double)]
lib.tf_drama3_speak_wav.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_float, C.c_int, C.c_float, C.c_uint64, C.c_int, C.POINTER(C.c_double)]
lib.tf_drama3_close.argtypes = [C.c_void_p]

lib_, model, codec, ref_wav, ref_text, text, prefix = sys.argv[1:8]
t = time.time()
h = lib.tf_drama3_open2(model.encode(), codec.encode(), 4096, 2048)
if not h:
    sys.exit(f"open: {lib.tf_drama3_error().decode()}")
print(f"open {time.time() - t:.1f} s")
codes = (C.c_int32 * (2048 * 10))()
nref = lib.tf_drama3_encode_wav(h, ref_wav.encode(), codes, 2048)
rows = (C.c_int32 * (4096 * 11))()
n = lib.tf_drama3_prompt(h, text.encode(), ref_text.encode(), codes, nref, rows, 4096)
print(f"reference {nref} frames, {n} prompt rows")
chunks = []
t0 = [0.0]


def on_audio(user, pcm, k):
    chunks.append((time.time() - t0[0], k, bytes(C.string_at(pcm, 4 * k))))


cb = CB(on_audio)
stats = (C.c_double * 6)()
t0[0] = time.time()
m = lib.tf_drama3_speak(h, rows, n, 1.0, 30, 0.9, 7, 1500, 4, 16, cb, None, None, stats)
print(f"speak: {m} frames, {len(chunks)} chunks, first at {chunks[0][0] * 1e3:.0f} ms (lib: first audio {stats[1]:.0f} ms), {stats[3]:.1f} frames/s, eos {stats[5]:.0f}")
import array
pcm = array.array("f")
for _, _, b in chunks:
    pcm.frombytes(b)
with wave.open(prefix + "_stream.wav", "wb") as w:
    w.setnchannels(1), w.setsampwidth(2), w.setframerate(44100)
    w.writeframes(array.array("h", (max(-32767, min(32767, round(x * 32767))) for x in pcm)).tobytes())
m2 = lib.tf_drama3_speak_wav(h, text.encode(), ref_wav.encode(), ref_text.encode(), (prefix + "_oneshot.wav").encode(), 1.0, 30, 0.9, 7, 1500, stats)
print(f"speak_wav: {m2} frames ({'same frames' if m2 == m else 'differs'}), first audio {stats[1]:.0f} ms")
lib.tf_drama3_close(h)
print("closed")
