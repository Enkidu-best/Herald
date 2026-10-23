# -*- coding: utf-8 -*-
"""Голос по языку отрывка (README, раздел 5).

    python3 "_общие модули/tts_client/tests/test_split.py"
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_client import split as ts  # noqa: E402

VOICES = {"en": "piper:en_US-joe-medium", "ru": "piper:ru_RU-denis-medium"}


class Split(unittest.TestCase):
    def test_answer_star(self):
        self.assertEqual(ts.segments("Answer: используйте STAR", "en"),
                         [("en", "Answer:"), ("ru", "используйте"), ("en", "STAR")])

    def test_russian_with_terms_numbers_quotes(self):
        self.assertEqual(ts.segments("Сократили срок на 30 % - по SLA «за 2 дня».", "ru"),
                         [("ru", "Сократили срок на 30 % - по"), ("en", "SLA «"), ("ru", "за 2 дня».")])
        self.assertEqual(ts.segments("2024 - 30 %", "ru"), [("ru", "2024 - 30 %")])
        self.assertEqual(ts.segments("«Hello», сказал он.", "ru"), [("en", "«Hello»,"), ("ru", "сказал он.")])

    def test_voice_never_gets_foreign_alphabet(self):
        calls = []

        def synth(text, voice):
            calls.append((voice, text))
            yield (22050, b"\0\0")
        tip = ("Formula: situation - task - result. Вы описали методологию, но не показали результат. "
               "Say: “I cut the time by 30%.”")
        list(ts.speak(tip, "en", VOICES.get, synth))
        for voice, text in calls:
            if voice.startswith("piper:en"):
                self.assertIsNone(re.search(r"[А-Яа-яЁё]", text), text)
            else:
                self.assertIsNone(re.search(r"[A-Za-z]", text), text)
        self.assertEqual([v for v, _ in calls], [VOICES["en"], VOICES["ru"], VOICES["en"]])

    def test_no_voice_for_language_skipped_not_spelled(self):
        calls, skipped = [], []
        list(ts.speak("Answer: используйте STAR", "en", {"en": "piper:en"}.get,
                      lambda t, v: (calls.append((v, t)), iter([]))[1], skipped))
        self.assertEqual(calls, [("piper:en", "Answer:"), ("piper:en", "STAR")])
        self.assertEqual(skipped, [("ru", "используйте")])

    def test_assert_clean(self):
        with self.assertRaises(ValueError):
            ts.assert_clean("Привет", "en")
        ts.assert_clean("Hello, 30%", "en")


if __name__ == "__main__":
    unittest.main(verbosity=2)
