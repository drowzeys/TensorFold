"""Clip preparation on synthetic speech: where it cuts, what it sets, and what it refuses."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

import drama_voice_clip as clip

RATE = clip.RATE


def word(f0: float, seconds: float, rate: int) -> np.ndarray:
    """A voiced syllable: eight harmonics under a Hann envelope, with half a semitone of vibrato."""
    t = np.arange(int(seconds * rate)) / rate
    phase = 2 * np.pi * np.cumsum(f0 * 2 ** (0.5 * np.sin(2 * np.pi * 4 * t) / 12)) / rate
    return np.hanning(len(t)) * sum(np.sin(k * phase) / k for k in range(1, 9))


def speech(f0s: list[float], rate: int = RATE, rests: tuple[float, ...] = (0.6,)) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """A six-word sentence per f0 after 0.4 s of quiet, each followed by the next of `rests`; and those rests, in s."""
    gap = 0.05
    parts, quiet, at = [np.zeros(int(0.4 * rate))], [], 0.4
    for i, f0 in enumerate(f0s):
        rest = rests[i % len(rests)]
        sentence = np.concatenate([np.concatenate((word(f0, 0.25, rate), np.zeros(int(gap * rate)))) for _ in range(6)])
        parts += [sentence, np.zeros(int(rest * rate))]
        at += len(sentence) / rate
        quiet.append((at - gap, at + rest))
        at += rest
    x = 0.3 * np.concatenate(parts)
    return x + 10 ** (-70 / 20) * np.random.default_rng(1).standard_normal(len(x)), quiet


class PrepareTests(unittest.TestCase):
    def refusal(self, x: np.ndarray) -> str:
        with self.assertRaises(SystemExit) as caught:
            clip.prepare(x, RATE)
        return str(caught.exception.code)

    def test_long_clip_is_cut_in_the_last_sentence_pause_that_fits_30_s(self):
        # a later comma rest also fits; the sentence rest before it wins
        x, rests = speech([120.0] * 20, rests=(0.2, 0.6))
        audio, report = clip.prepare(x, RATE)
        self.assertGreater(report["seconds"], 30 - 4.3)
        self.assertLessEqual(report["seconds"], 30)
        self.assertAlmostEqual(len(audio) / RATE, report["seconds"], places=2)
        cut_at = report["source_seconds"][1]
        self.assertTrue(any(start < cut_at < end and end - start > 0.5 for start, end in rests), cut_at)
        self.assertGreater(report["speech_left_out_seconds"], 0)
        self.assertLess(20 * np.log10(np.sqrt(np.mean(audio[-RATE // 20 :] ** 2))), report["level_dbfs"] - clip.QUIET_DB)

    def test_level_is_the_same_whatever_the_recording_gain_and_peaks_stay_under_minus_1_dbfs(self):
        x, _ = speech([120.0] * 6)
        quiet, report = clip.prepare(0.02 * x, RATE)
        loud, _ = clip.prepare(2.5 * x, RATE)
        self.assertEqual(report["level_dbfs"], -20.0)
        np.testing.assert_allclose(quiet, loud, atol=1e-6)
        self.assertLessEqual(np.max(np.abs(loud)), 10 ** (-1 / 20) + 1e-9)
        clicked = x.copy()
        clicked[RATE] = 3.0
        audio, report = clip.prepare(clicked, RATE)
        self.assertAlmostEqual(np.max(np.abs(audio)), 10 ** (-1 / 20), places=6)
        self.assertLess(report["level_dbfs"], -20.0)

    def test_stereo_48_khz_becomes_mono_44_1_khz_at_the_same_pitch(self):
        x, _ = speech([150.0] * 6, rate=48000)
        audio, report = clip.prepare(np.stack((x, 0.5 * x), axis=1), 48000)
        _, native = clip.prepare(speech([150.0] * 6)[0], RATE)
        self.assertEqual(audio.ndim, 1)
        self.assertAlmostEqual(report["seconds"], native["seconds"], delta=0.02)
        self.assertAlmostEqual(np.nanmedian(clip.pitch(audio)), 150.0, delta=3.0)

    def test_a_short_clip_is_kept_whole_from_a_quarter_second_before_its_first_word(self):
        x, rests = speech([120.0] * 6)
        audio, report = clip.prepare(x, RATE)
        self.assertEqual(report["speech_left_out_seconds"], 0)
        self.assertAlmostEqual(report["source_seconds"][0], 0.4 - 0.25, delta=0.05)
        self.assertLessEqual(report["source_seconds"][1], rests[-1][0] + 0.25 + 0.1)
        self.assertLess(20 * np.log10(np.sqrt(np.mean(audio[: RATE // 5] ** 2))), -20.0 - 25.0)

    def test_clipping_is_refused_and_a_peak_at_full_scale_is_not(self):
        x, _ = speech([120.0] * 6)
        self.assertIn("clipped", self.refusal(np.clip(5 * x, -1, 1)))
        clip.prepare(x / np.max(np.abs(x)), RATE)

    def test_noise_is_refused(self):
        x, _ = speech([120.0] * 6)
        noisy = x + 0.03 * np.random.default_rng(2).standard_normal(len(x))
        self.assertIn("background", self.refusal(noisy))

    def test_noise_is_judged_on_the_kept_clip_not_on_quiet_that_is_cut_away(self):
        x, _ = speech([120.0] * 6)
        noise = 0.01 * np.random.default_rng(10).standard_normal(len(x))
        noise[: int(0.4 * RATE)] = 0
        tail = 1e-5 * np.random.default_rng(11).standard_normal(20 * RATE)
        self.assertIn("background", self.refusal(x + noise))
        self.assertIn("background", self.refusal(np.concatenate((x + noise, tail))))
        _, clean = clip.prepare(x, RATE)
        _, padded = clip.prepare(np.concatenate((x, tail)), RATE)
        self.assertAlmostEqual(padded["snr_db"], clean["snr_db"], delta=0.5)

    def test_music_under_the_voice_is_refused_though_it_is_quiet(self):
        x, _ = speech([120.0] * 6)
        t = np.arange(len(x)) / RATE
        chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6, 440.0))
        self.assertIn("music or hum", self.refusal(x + 0.0015 * chord))

    def test_two_speakers_are_refused_and_one_speakers_intonation_is_not(self):
        self.assertIn("more than one speaker", self.refusal(speech([110.0, 220.0] * 4)[0]))
        _, report = clip.prepare(speech([110.0, 140.0, 125.0, 115.0, 135.0, 120.0])[0], RATE)
        self.assertEqual(report["speech_stretches"], 6)
        self.assertLess(report["pitch_spread_semitones"], clip.MAX_SPREAD_ST)

    def test_too_short_uncuttable_and_mid_word_clips_are_refused(self):
        self.assertIn("a voice needs 10 s", self.refusal(speech([120.0] * 3)[0]))
        self.assertIn("no pause", self.refusal(speech([120.0] * 20, rests=(0.05,))[0]))
        x, _ = speech([120.0] * 6)
        self.assertIn("starts inside speech", self.refusal(x[int(0.5 * RATE) :]))
        for empty in (np.zeros(0), np.full(100, 0.01), np.zeros((0, 2))):
            self.assertIn("a voice needs 10 s", self.refusal(empty))


class TranscriptTests(unittest.TestCase):
    def run_main(self, folder: Path, x: np.ndarray, *words: str) -> None:
        sf.write(folder / "take.wav", x, RATE, subtype="PCM_16")
        with patch("sys.argv", ["drama_voice_clip.py", str(folder / "take.wav"), "--out", str(folder / "me.wav"), *words]):
            with contextlib.redirect_stdout(io.StringIO()):
                clip.main()

    def test_a_given_transcript_is_kept_only_when_the_cut_keeps_all_the_speech(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "take.txt").write_text("  Every word of the take.\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "leaves speech out"):
                self.run_main(folder, speech([120.0] * 20)[0], "--text", str(folder / "take.txt"))
            self.assertFalse((folder / "me.txt").exists())
            self.run_main(folder, speech([120.0] * 6)[0], "--text", str(folder / "take.txt"))
            self.assertEqual((folder / "me.txt").read_text(encoding="utf-8"), "Every word of the take.\n")
            self.assertEqual(sf.info(folder / "me.wav").samplerate, RATE)

    def test_whisper_hears_the_cut_clip_and_leaves_a_draft_to_confirm(self):
        heard = []
        with tempfile.TemporaryDirectory() as name, patch.object(clip, "transcribe", lambda audio, _: heard.append(audio) or "draft"):
            folder = Path(name)
            self.run_main(folder, speech([120.0] * 20)[0], "--asr", "whisper")
            self.assertEqual((folder / "me.draft.txt").read_text(encoding="utf-8"), "draft\n")
            self.assertFalse((folder / "me.txt").exists())
            self.assertEqual(len(heard[0]), sf.info(folder / "me.wav").frames)

    def test_whisper_never_replaces_a_clip_whose_transcript_was_confirmed(self):
        with tempfile.TemporaryDirectory() as name, patch.object(clip, "transcribe", lambda *_: "new words"):
            folder = Path(name)
            (folder / "me.txt").write_text("Old words.\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "me.txt holds a confirmed transcript"):
                self.run_main(folder, speech([120.0] * 6)[0], "--asr", "whisper")
            self.assertFalse((folder / "me.wav").exists() or (folder / "me.draft.txt").exists())
            self.assertEqual((folder / "me.txt").read_text(encoding="utf-8"), "Old words.\n")

    def test_a_recording_too_short_to_measure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "take.txt").write_text("Hi.\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "a voice needs 10 s"):
                self.run_main(folder, np.full(100, 0.01), "--text", str(folder / "take.txt"))
            self.assertEqual(sorted(path.name for path in folder.iterdir()), ["take.txt", "take.wav"])


if __name__ == "__main__":
    unittest.main()
