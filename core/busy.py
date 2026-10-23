"""Кто занимает видеокарту (3.7): общий замок проектов Андрея и падение скорости.

Программы Андрея перед тяжёлой работой (синтез, распознавание, модели) кладут
папку `.heavy-lock` с файлом `owner` («Понятный документ Sat Oct 17 …») рядом с
папками проектов. Пока её держит другая программа, видеокарта делится - озвучка
идёт медленнее. На компьютере без таких программ замка нет - тогда о занятости
говорит только падение скорости (окно сравнивает недавнюю скорость с начальной).
"""

from __future__ import annotations

from pathlib import Path

LOCK_NAME = ".heavy-lock"
SLOWDOWN = 1.5          # во сколько раз упала скорость, чтобы сказать «занята»


def lock_path() -> Path:
    from .heavylock import lock_path as _lp
    return _lp()


def lock_owner() -> str | None:
    """Кто держит замок, или None (свободен или держит ЭТОТ Herald - задание 13 Г1).
    Другой процесс Herald (ночная проверка, второе окно) - тоже «занята»: его имя."""
    from . import heavylock
    p = lock_path()
    if not p.is_dir() or heavylock.ours(p):
        return None
    who = heavylock.owner(p)
    return "другая программа" if who in ("", "?") else who


def lock_free() -> bool:
    return lock_owner() is None


def slowed(recent_rate: float | None, base_rate: float | None) -> bool:
    """Скорость (доля работы в секунду) упала больше чем в SLOWDOWN раз."""
    return bool(recent_rate and base_rate and base_rate / max(recent_rate, 1e-9) > SLOWDOWN)


__all__ = ["lock_path", "lock_owner", "lock_free", "slowed", "SLOWDOWN"]
