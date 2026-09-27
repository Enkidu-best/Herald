"""Реестр движков синтеза.

Добавить новый движок = положить файл рядом и дописать одну строку в ENGINES.
"""

from __future__ import annotations

from .base import TTSEngine, Voice
from .silero import SileroEngine
from .f5 import F5Engine
from .f5_mlx import F5MLXEngine
from .apple import AppleEngine

# id -> (человекочитаемое имя, класс)
ENGINES: dict[str, tuple[str, type[TTSEngine]]] = {
    "f5mlx": ("F5-MLX — тот же голос, быстрый на Apple Silicon (рекомендуется)", F5MLXEngine),
    "f5": ("F5-TTS (torch) — тот же голос, но медленный на CPU", F5Engine),
    "apple": ("Голос macOS — мгновенно, но готовый голос, не клон", AppleEngine),
    "silero": ("Silero — ровный диктор, очень быстрый (запасной)", SileroEngine),
}


def list_engines() -> list[tuple[str, str]]:
    """[(id, человекочитаемое имя), ...] для выпадающего списка в GUI."""
    return [(eid, title) for eid, (title, _cls) in ENGINES.items()]


def get_engine(engine_id: str, **kw) -> TTSEngine:
    if engine_id not in ENGINES:
        raise ValueError(f"Неизвестный движок: {engine_id}. Доступны: {list(ENGINES)}")
    _title, cls = ENGINES[engine_id]
    return cls(**kw)


__all__ = ["TTSEngine", "Voice", "get_engine", "list_engines", "ENGINES"]
