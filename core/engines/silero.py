"""Движок синтеза на модели Silero (v4, русский).

Silero — быстрый и очень стабильный на длинном тексте движок. Работает на CPU,
даёт ровный «дикторский» голос. Модель — один файл ~40 МБ, скачивается один раз
и кэшируется; после этого всё офлайн.

Голоса v4_ru: aidar (м), eugene (м), baya (ж), kseniya (ж), xenia (ж).
"""

from __future__ import annotations

import os
import re
import numpy as np

from .base import TTSEngine, Voice

# Официальный адрес модели. Скачиваем сами (torch.hub ходит на GitHub, а он
# в песочнице бывает недоступен через прокси — прямая ссылка надёжнее).
MODEL_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"
MODEL_FILENAME = "v4_ru.pt"

VOICES = [
    Voice("eugene", "Евгений (муж.)", "м"),
    Voice("aidar", "Айдар (муж.)", "м"),
    Voice("baya", "Бая (жен.)", "ж"),
    Voice("kseniya", "Ксения (жен.)", "ж"),
    Voice("xenia", "Ксения-2 (жен.)", "ж"),
]


def default_model_dir() -> str:
    """Каталог кэша модели: рядом с моделями Whisper от Scribe, в домашней папке."""
    base = os.path.join(os.path.expanduser("~"), ".cache", "text2audio", "silero")
    os.makedirs(base, exist_ok=True)
    return base


class SileroEngine(TTSEngine):
    name = "silero"
    sample_rate = 48000
    max_chunk_chars = 800          # v4 надёжно тянет ~1000, берём с запасом
    supports_cloning = False
    stress_format = "+"            # Silero понимает '+' перед ударной гласной

    def __init__(self, model_dir: str | None = None, threads: int = 4) -> None:
        super().__init__()
        self.model_dir = model_dir or default_model_dir()
        self.threads = threads
        self._model = None

    def list_voices(self) -> list[Voice]:
        return list(VOICES)

    # --- загрузка -------------------------------------------------------
    def _model_path(self) -> str:
        return os.path.join(self.model_dir, MODEL_FILENAME)

    def _download(self, progress=None) -> None:
        import urllib.request
        import ssl
        # python.org Python на macOS не видит системные сертификаты — берём
        # набор из certifi, иначе скачивание падает с CERTIFICATE_VERIFY_FAILED.
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
        path = self._model_path()
        tmp = path + ".part"
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "text2audio"})
        with urllib.request.urlopen(req, timeout=120, context=ctx) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                buf = r.read(1 << 16)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress and total:
                    progress(done / total)
        os.replace(tmp, path)

    def load(self, progress=None) -> None:
        import torch
        torch.set_num_threads(self.threads)
        path = self._model_path()
        if not os.path.isfile(path):
            self._download(progress=progress)
        self._model = torch.package.PackageImporter(path).load_pickle("tts_models", "model")
        self._model.to("cpu")
        self._loaded = True

    def available_speaker_ids(self) -> list[str]:
        if self._model is None:
            return [v.id for v in VOICES]
        return list(getattr(self._model, "speakers", [v.id for v in VOICES]))

    # --- синтез ---------------------------------------------------------
    def synth_chunk(self, text: str, voice: str, **kw) -> np.ndarray:
        self.ensure_loaded()
        if not self.has_speech(text):
            return np.zeros(0, dtype=np.float32)
        # Silero падает на слишком длинном куске — подстрахуемся жёстким лимитом.
        text = text.strip()
        if len(text) > 1000:
            text = text[:1000]
        for attempt in (text, self._sanitize(text)):
            if not self.has_speech(attempt):
                break
            try:
                audio = self._model.apply_tts(
                    text=attempt,
                    speaker=voice,
                    sample_rate=self.sample_rate,
                    put_accent=True,
                    put_yo=True,
                )
                return audio.numpy().astype(np.float32)
            except Exception:
                # один «неудобный» кусок не должен ронять всю книгу:
                # пробуем очищенный вариант, иначе тихо пропускаем кусок
                continue
        return np.zeros(0, dtype=np.float32)

    @staticmethod
    def _sanitize(text: str) -> str:
        """Оставить только то, что Silero точно проговорит (буквы, цифры, базовая пунктуация)."""
        keep = re.sub(r"[^0-9A-Za-zА-Яа-яЁё \-.,!?;:—\n]", " ", text)
        return re.sub(r"\s{2,}", " ", keep).strip()
