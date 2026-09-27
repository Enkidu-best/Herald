"""Единый интерфейс движка синтеза речи.

Любой движок (Silero, Chatterbox, ...) реализует этот интерфейс, а пайплайн
работает с ними одинаково. Движок отвечает только за «текст -> звук одного
куска». Нарезку на куски, паузы и склейку делает пайплайн — так правила
чтения книг не зависят от конкретной модели.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Voice:
    """Голос движка. id — то, что передаётся в synth; title — что видит человек."""
    id: str
    title: str
    gender: str = ""          # "м" / "ж" / ""
    language: str = "ru"


class TTSEngine:
    """Базовый класс движка. Наследники переопределяют load() и synth_chunk()."""

    name: str = "base"
    sample_rate: int = 48000
    # Максимальная длина одного куска в символах. Пайплайн режет текст под неё.
    max_chunk_chars: int = 800
    # Поддерживает ли движок клонирование голоса по образцу.
    supports_cloning: bool = False
    # Формат ударения, который движок понимает: "+" (перед гласной) или None.
    # Пайплайн по этому полю решает, звать ли RUAccent перед синтезом.
    stress_format: str | None = None

    def __init__(self) -> None:
        self._loaded = False

    # --- голоса ---------------------------------------------------------
    def list_voices(self) -> list[Voice]:
        """Список доступных встроенных голосов."""
        raise NotImplementedError

    # --- жизненный цикл -------------------------------------------------
    def load(self) -> None:
        """Загрузить модель в память (может скачать её при первом запуске)."""
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    # --- синтез ---------------------------------------------------------
    def synth_chunk(self, text: str, voice: str, **kw) -> np.ndarray:
        """Озвучить один короткий кусок. Вернуть float32 моно [-1..1] на sample_rate.

        text уже нормализован и по длине укладывается в max_chunk_chars.
        """
        raise NotImplementedError

    # Удобный признак «в тексте есть что произносить».
    @staticmethod
    def has_speech(text: str) -> bool:
        return any(ch.isalpha() for ch in text)
