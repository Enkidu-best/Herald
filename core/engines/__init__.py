"""Реестр движков синтеза.

Добавить новый движок = положить файл рядом и дописать одну строку в ENGINES.
"""

from __future__ import annotations

from .base import TTSEngine, Voice
from .silero import SileroEngine
from .f5 import F5Engine
from .f5_mlx import F5MLXEngine
from .f5_torch import F5TorchEngine
from .apple import AppleEngine
from .fast import VoskEngine, PiperEngine

# id -> (человекочитаемое имя, класс)
ENGINES: dict[str, tuple[str, type[TTSEngine]]] = {
    "f5mlx": ("F5-MLX — тот же голос, быстрый на Apple Silicon (рекомендуется)", F5MLXEngine),
    "f5torch": ("F5 на torch — тот же голос и алгоритм; видеокарта NVIDIA или процессор",
                F5TorchEngine),
    "f5": ("F5-TTS (torch, через f5_tts.api) — прежний путь, для сверки", F5Engine),
    "apple": ("Голос macOS — мгновенно, но готовый голос, не клон", AppleEngine),
    "silero": ("Silero — ровный диктор, очень быстрый (запасной)", SileroEngine),
    "vosk": ("Vosk TTS — быстрые русские голоса (3.6, по выбору Андрея)", VoskEngine),
    "piper": ("Piper — быстрые голоса (3.6, по выбору Андрея)", PiperEngine),
}


def apple_silicon() -> bool:
    import platform
    import sys
    return sys.platform == "darwin" and platform.machine() == "arm64"


def clone_engine() -> str:
    """Каким движком клонировать голос на ЭТОМ компьютере - без выбора у пользователя.

    Apple Silicon - MLX (видеоядро Apple); иначе F5 на torch: видеокарта NVIDIA,
    если есть, или процессор (это медленно - окно предупреждает). Переменная
    HERALD_ENGINE=f5torch|f5mlx - для проверок.
    """
    import os
    want = os.environ.get("HERALD_ENGINE")
    if want in ENGINES:
        return want
    if apple_silicon():
        try:
            import mlx.core  # noqa: F401
            return "f5mlx"
        except Exception:
            pass                      # нет MLX - падать не надо, есть torch
    return "f5torch"


def accel_label() -> str:
    """Чем считается озвучка: для строки в окне и для прогноза времени."""
    eng = clone_engine()
    if eng == "f5mlx":
        return "mlx"
    from .f5_torch import torch_device
    return "cuda" if torch_device() == "cuda" else "cpu"


def list_engines() -> list[tuple[str, str]]:
    """[(id, человекочитаемое имя), ...] для выпадающего списка в GUI."""
    return [(eid, title) for eid, (title, _cls) in ENGINES.items()]


def get_engine(engine_id: str, **kw) -> TTSEngine:
    if engine_id not in ENGINES:
        raise ValueError(f"Неизвестный движок: {engine_id}. Доступны: {list(ENGINES)}")
    _title, cls = ENGINES[engine_id]
    return cls(**kw)


__all__ = ["TTSEngine", "Voice", "get_engine", "list_engines", "ENGINES", "clone_engine",
           "accel_label", "apple_silicon"]
