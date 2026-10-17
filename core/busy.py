"""Кто занимает видеокарту (3.7): общий замок проектов Андрея и падение скорости.

Программы Андрея перед тяжёлой работой (синтез, распознавание, модели) кладут
папку `.heavy-lock` с файлом `owner` («Понятный документ Sat Oct 17 …») рядом с
папками проектов. Пока её держит другая программа, видеокарта делится - озвучка
идёт медленнее. На компьютере без таких программ замка нет - тогда о занятости
говорит только падение скорости (окно сравнивает недавнюю скорость с начальной).
"""

from __future__ import annotations

import os
from pathlib import Path

LOCK_NAME = ".heavy-lock"
SLOWDOWN = 1.5          # во сколько раз упала скорость, чтобы сказать «занята»


def lock_path() -> Path:
    here = Path(__file__).resolve().parent.parent          # папка Herald
    return here.parent / LOCK_NAME


def lock_owner() -> str | None:
    """Имя программы, которая держит замок, или None (свободен / держит Herald)."""
    p = lock_path()
    if not p.is_dir():
        return None
    try:
        text = (p / "owner").read_text(encoding="utf-8").strip()
    except OSError:
        return "другая программа"
    name = text.split(" Mon ")[0].split(" Tue ")[0].split(" Wed ")[0].split(" Thu ")[0]
    for d in (" Fri ", " Sat ", " Sun "):
        name = name.split(d)[0]
    name = name.strip() or "другая программа"
    return None if name.lower().startswith("herald") else name


def lock_free() -> bool:
    return not lock_path().is_dir() or lock_owner() is None


def slowed(recent_rate: float | None, base_rate: float | None) -> bool:
    """Скорость (доля работы в секунду) упала больше чем в SLOWDOWN раз."""
    return bool(recent_rate and base_rate and base_rate / max(recent_rate, 1e-9) > SLOWDOWN)


__all__ = ["lock_path", "lock_owner", "lock_free", "slowed", "SLOWDOWN"]
_ = os
