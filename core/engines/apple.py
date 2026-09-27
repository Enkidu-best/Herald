"""Встроенный в macOS синтез (голоса системы) — самый быстрый запасной вариант.

Считает примерно в 30 раз быстрее реального времени (час звука за пару минут) и
не требует вообще никаких моделей: всё уже в системе. Расплата — это ГОТОВЫЙ
голос, а не клон диктора, и звучит он заметно проще, чем F5.

Ударения не поддерживает: у системных голосов свой словарь, символ «+» они бы
прочитали вслух. Поэтому stress_format = None, и пайплайн для этого движка
RUAccent не зовёт.

Больше голосов можно доустановить: Системные настройки -> Универсальный доступ
-> Устная речь -> Системный голос -> Управление голосами -> Русский.
Из коробки в macOS стоит только базовый compact-вариант «Milena».
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from .base import TTSEngine, Voice

SAMPLE_RATE = 24000
DEFAULT_WPM = 180             # слов в минуту; у `say` по умолчанию ~175


class AppleEngine(TTSEngine):
    name = "apple"
    sample_rate = SAMPLE_RATE
    max_chunk_chars = 1500     # системному синтезу длина куска безразлична
    supports_cloning = False
    stress_format = None       # «+» системный голос прочитал бы как «плюс»

    def __init__(self, speed: float = 1.0, language: str = "ru") -> None:
        super().__init__()
        self.speed = speed
        self.language = language

    # --- голоса ---------------------------------------------------------
    def list_voices(self) -> list[Voice]:
        say = shutil.which("say")
        if not say:
            return []
        out = subprocess.run([say, "-v", "?"], capture_output=True, text=True).stdout
        voices = []
        for line in out.splitlines():
            m = re.match(r"^(.+?)\s{2,}([a-z]{2}[_-][A-Z]{2})\s", line)
            if m and m.group(2).lower().startswith(self.language):
                voices.append(Voice(m.group(1).strip(), m.group(1).strip(), "", self.language))
        return voices

    def load(self, progress=None) -> None:
        if not shutil.which("say"):
            raise RuntimeError("Системный синтез `say` доступен только на macOS.")
        self._loaded = True

    # --- синтез ---------------------------------------------------------
    def synth_chunk(self, text: str, voice: str, speed_mul: float = 1.0,
                    **kw) -> np.ndarray:
        self.ensure_loaded()
        if not self.has_speech(text):
            return np.zeros(0, dtype=np.float32)
        import soundfile as sf

        tmpdir = tempfile.mkdtemp(prefix="t2a_say_")
        txt_path = os.path.join(tmpdir, "in.txt")
        aiff = os.path.join(tmpdir, "out.aiff")
        wav = os.path.join(tmpdir, "out.wav")
        try:
            # текст передаём файлом: через аргумент длинная строка упирается в
            # ограничение командной строки, а кавычки внутри книги ломают вызов
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text.strip())
            cmd = ["say", "-f", txt_path, "-o", aiff]
            if voice:
                cmd[1:1] = ["-v", voice]
            rate = self.speed * speed_mul
            if abs(rate - 1.0) > 0.01:
                cmd[1:1] = ["-r", str(int(DEFAULT_WPM * rate))]
            subprocess.run(cmd, check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                            "-i", aiff, "-ac", "1", "-ar", str(SAMPLE_RATE), wav],
                           check=True)
            audio, _sr = sf.read(wav)
            return np.asarray(audio, dtype=np.float32).reshape(-1)
        except Exception as e:
            import sys
            print(f"[Apple] пропущен кусок: {e}", file=sys.stderr)
            return np.zeros(0, dtype=np.float32)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
