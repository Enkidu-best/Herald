"""Дополнительные быстрые голоса (3.6): Vosk TTS и Piper.

Отбирает Андрей на слух (слепой набор «Сравнение 3.6 - быстрые голоса - …»);
в программе появляются только голоса из ENABLED. Пакеты (vosk-tts, piper-tts)
необязательны: нет пакета - голоса нет, остальное работает.

Лицензии (карточки моделей, 16.10.2026):
  * Vosk TTS ru 0.9 multi - Apache-2.0 (движок vosk-tts - Apache-2.0);
  * Piper ru_RU-denis, en_US-joe - CC0; en_US-kristin, en_GB-cori - общественное
    достояние; сам движок piper-tts 1.8 - GPL-3.0 (для установщика Windows это
    важно: включать его в сборку - только с соблюдением GPL).
Голоса Piper с некоммерческой или неясной лицензией (ruslan, irina, ryan, hfc,
lessac, amy, alan) в набор не вошли.
"""

from __future__ import annotations

import os

import numpy as np

from .base import TTSEngine, Voice

SR = 24000          # общий выход: у Piper 22 050 Гц, у Vosk 22 050 Гц - приводим

# (id, имя, пол, язык, движок, параметр модели)
CANDIDATES = [
    ("vosk:0", "Vosk 0", "ж", "ru", "vosk", 0),
    ("vosk:1", "Vosk 1", "ж", "ru", "vosk", 1),
    ("vosk:2", "Vosk 2", "ж", "ru", "vosk", 2),
    ("vosk:3", "Vosk 3", "м", "ru", "vosk", 3),
    ("vosk:4", "Vosk 4", "м", "ru", "vosk", 4),
    ("piper:ru_RU-denis-medium", "Денис", "м", "ru", "piper", "ru_RU-denis-medium"),
    ("piper:en_US-joe-medium", "Joe", "м", "en", "piper", "en_US-joe-medium"),
    ("piper:en_US-kristin-medium", "Kristin", "ж", "en", "piper", "en_US-kristin-medium"),
    ("piper:en_GB-cori-medium", "Cori", "ж", "en", "piper", "en_GB-cori-medium"),
]
# Голоса, отмеченные Андреем в слепом наборе 3.6 (до оценки - пусто)
ENABLED: set[str] = set()
VOSK_MODEL = "vosk-model-tts-ru-0.9-multi"


def enabled_voices() -> list[Voice]:
    """Включённые голоса, движок которых установлен."""
    out = []
    for vid, title, g, lang, eng, _p in CANDIDATES:
        if vid in ENABLED and _available(eng):
            out.append(Voice(vid, title, g, lang))
    return out


def engine_of(voice_id: str) -> str | None:
    return next((c[4] for c in CANDIDATES if c[0] == voice_id), None)


def _available(eng: str) -> bool:
    import importlib.util
    return importlib.util.find_spec({"vosk": "vosk_tts", "piper": "piper"}[eng]) is not None


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return a.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, SR)
    return resample_poly(a, SR // g, sr // g).astype(np.float32)


class VoskEngine(TTSEngine):
    """Vosk TTS: русская модель с пятью дикторами, на процессоре, ~0,2 RTF."""
    name = "vosk"
    sample_rate = SR
    max_chunk_chars = 400
    stress_format = "+"           # «+» перед ударной гласной модель понимает

    def __init__(self) -> None:
        super().__init__()
        self._synth = None

    def list_voices(self) -> list[Voice]:
        return [v for v in enabled_voices() if engine_of(v.id) == "vosk"]

    def cache_key(self) -> str:
        return f"vosk|{VOSK_MODEL}"

    def load(self) -> None:
        try:
            import certifi                      # python.org Python на Mac: сертификаты
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass
        from vosk_tts import Model, Synth
        model = Model(model_name=VOSK_MODEL)
        self._sr = int((getattr(model, "config", None) or {}).get("audio", {})
                       .get("sample_rate") or 22050)
        self._synth = Synth(model)

    def synth_chunk(self, text: str, voice: str, speed_mul: float = 1.0, **kw) -> np.ndarray:
        self.ensure_loaded()
        spk = int(voice.split(":", 1)[1])
        out = np.asarray(self._synth.synth_audio(text, speaker_id=spk))
        a = out.astype(np.float32) / 32768.0 if out.dtype == np.int16 else out.astype(np.float32)
        return _resample(a, self._sr)


class PiperEngine(TTSEngine):
    """Piper: голос - отдельная модель ONNX (~60 МБ), на процессоре, ~0,03 RTF."""
    name = "piper"
    sample_rate = SR
    max_chunk_chars = 400
    stress_format = None          # ударения ставит сам (espeak-ng)

    def __init__(self) -> None:
        super().__init__()
        self._voices: dict = {}

    def list_voices(self) -> list[Voice]:
        return [v for v in enabled_voices() if engine_of(v.id) == "piper"]

    def cache_key(self) -> str:
        return "piper|1.8"

    def load(self) -> None:
        import piper  # noqa: F401 - нет пакета - пусть упадёт здесь

    def _voice(self, name: str):
        if name not in self._voices:
            from huggingface_hub import hf_hub_download
            from piper import PiperVoice
            lang, who, q = name.split("-")
            base = f"{lang.split('_')[0]}/{lang}/{who}/{q}/{name}"
            self._voices[name] = PiperVoice.load(
                hf_hub_download("rhasspy/piper-voices", base + ".onnx"),
                config_path=hf_hub_download("rhasspy/piper-voices", base + ".onnx.json"))
        return self._voices[name]

    def synth_chunk(self, text: str, voice: str, speed_mul: float = 1.0, **kw) -> np.ndarray:
        self.ensure_loaded()
        v = self._voice(voice.split(":", 1)[1])
        a = np.concatenate([c.audio_float_array for c in v.synthesize(text)] or
                           [np.zeros(0, np.float32)])
        return _resample(a, v.config.sample_rate)
