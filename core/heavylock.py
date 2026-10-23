# -*- coding: utf-8 -*-
"""Общий замок видеокарты `../.heavy-lock` - общий модуль `common/lock` («Разработка ПО/_общие модули/lock», описание -
README.md там же; копию в common/ не править). Здесь - только привязка к Herald с прежним API (стандарт 6.7):
HeavyLock() / try_acquire / acquire(cancel, on_wait(кто), wait) / release, lock_path(), owner(), owner_pid(), ours(),
held_outside().

  * замок держит книга; «Пауза» дольше 15 минут - отпускается сам, «Продолжить» берёт снова (стандарт 2.5);
  * HERALD_HEAVY_LOCK_HELD=1 (или HEAVY_LOCK_HELD=1) - замок уже держит ночная цепочка, которая запустила Herald;
  * HERALD_HEAVY_LOCK=<путь> (или HEAVY_LOCK) - другой путь замка (проверки не трогают настоящую очередь).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from common.lock import heavy_lock as hl

OWNER = "Herald"


def lock_path() -> Path:
    return hl.default_path("HERALD_HEAVY_LOCK")


def held_outside() -> bool:
    """Замок держит тот, кто запустил Herald (ночная цепочка)."""
    return os.environ.get("HERALD_HEAVY_LOCK_HELD") == "1" or os.environ.get("HEAVY_LOCK_HELD") == "1"


def owner_pid(lock: Path | None = None) -> int | None:
    h = hl.status(lock or lock_path())
    return h.pid if h else None


def owner(lock: Path | None = None) -> str:
    """Кто держит замок: «Понятный документ», «Herald (12 проверки)»… Пусто - свободен."""
    h = hl.status(lock or lock_path())
    return (h.program or "?") if h else ""


def ours(lock: Path | None = None) -> bool:
    """Замок держит этот процесс Herald (или ночная цепочка, которая его запустила)."""
    return held_outside() or owner_pid(lock) == os.getpid()


class HeavyLock(hl.HeavyLock):
    """Замок на время книги: try_acquire / acquire (ждать, с отменой) / release, простой на паузе - idle_check."""

    def __init__(self, path: Path | None = None, what: str = "озвучка книги") -> None:
        super().__init__(OWNER, what, path=path, held_env="HERALD_HEAVY_LOCK_HELD", path_env="HERALD_HEAVY_LOCK")

    def acquire(self, cancel: threading.Event | None = None, on_wait=None, wait: float | None = 7 * 86400,
                poll: float = 2.0) -> bool:
        """Ждать свободного замка; on_wait(кто держит) - раз в poll с (on_wait.full - (holder, ahead)). Отмена - False."""
        cb = None
        if on_wait is not None:
            full = getattr(on_wait, "full", False)
            cb = (lambda h, a: on_wait(h, a)) if full else (lambda h, a: on_wait(h.program if h else ""))
        return super().acquire(cancel=cancel, on_wait=cb, wait=wait, poll=poll)


__all__ = ["HeavyLock", "lock_path", "owner", "owner_pid", "ours", "held_outside", "OWNER"]
