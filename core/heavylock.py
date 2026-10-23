# -*- coding: utf-8 -*-
"""Общий замок видеокарты `../.heavy-lock` - Herald его БЕРЁТ на озвучку книги (задание 13 Г1,
стандарт 6.7). Копия эталона Interview Coach `coach/lock.py` (общего модуля
`_общие модули/` ещё нет): папка `.heavy-lock` рядом с папками проектов, в ней `owner` -
«Herald <дата> pid <номер>»; взять - mkdir, отдать - удалить, если замок наш.

  * замок от упавшего Herald (номер процесса не жив) снимается сразу, чужой - только
    если старше 3 часов и его процесс не жив (живого владельца по возрасту не снимаем:
    ошибка Herald 21.10);
  * HERALD_HEAVY_LOCK_HELD=1 - замок уже держит ночная цепочка, которая запустила
    Herald: повторно не берём и не отдаём;
  * HERALD_HEAVY_LOCK=<путь> - другой путь замка (проверки не трогают настоящую очередь).
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

OWNER = "Herald"


def lock_path() -> Path:
    env = os.environ.get("HERALD_HEAVY_LOCK")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent / ".heavy-lock"


def held_outside() -> bool:
    """Замок держит тот, кто запустил Herald (ночная цепочка)."""
    return os.environ.get("HERALD_HEAVY_LOCK_HELD") == "1"


def _alive(pid: int) -> bool:
    """Жив ли процесс. На Windows os.kill(pid, 0) - Ctrl+C всей группе, поэтому через WinAPI."""
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259                 # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _text(lock: Path) -> str | None:
    try:
        return (lock / "owner").read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def owner_pid(lock: Path | None = None) -> int | None:
    m = re.search(r"pid (\d+)", _text(lock or lock_path()) or "")
    return int(m.group(1)) if m else None


def owner(lock: Path | None = None) -> str:
    """Кто держит замок: «Понятный документ», «Herald (12 проверки)»… Пусто - свободен."""
    lock = lock or lock_path()
    txt = _text(lock)
    if txt is None:
        return "?" if lock.exists() else ""
    m = re.match(r"(.+?)\s+(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", txt)
    return (m.group(1) if m else txt.split(" pid ")[0]).strip() or "?"


def ours(lock: Path | None = None) -> bool:
    """Замок держит этот процесс Herald (или ночная цепочка, которая его запустила)."""
    return held_outside() or owner_pid(lock) == os.getpid()


def _stale(lock: Path) -> bool:
    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False
    txt = _text(lock) or ""
    pid = owner_pid(lock)
    dead = not (pid and _alive(pid))
    if dead and txt.startswith(OWNER + " "):
        return True          # свой замок от упавшего процесса - снимаем сразу, не ждём 3 часа
    return age > 3 * 3600 and dead


class HeavyLock:
    """Замок на время книги: try_acquire / acquire (ждать, с отменой) / release."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self.held = False
        self._mu = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path or lock_path()

    def try_acquire(self) -> bool:
        if held_outside():
            self.held = True
            return True
        with self._mu:
            if self.held:
                return True
            for _ in range(2):          # вторая попытка - после снятия брошенного замка
                try:
                    self.path.mkdir()
                except FileExistsError:
                    if _stale(self.path):
                        shutil.rmtree(self.path, ignore_errors=True)
                        continue
                    return False
                except OSError:
                    return False        # папки проектов нет или нет прав - работаем без замка
                (self.path / "owner").write_text(
                    f"{OWNER} {dt.datetime.now():%a %b %d %H:%M:%S %Y} pid {os.getpid()}\n",
                    encoding="utf-8")
                self.held = True
                return True
            return False

    def acquire(self, cancel: threading.Event | None = None, on_wait=None, wait: float = 7 * 86400) -> bool:
        """Ждать свободного замка; on_wait(кто держит) - раз в ~2 с. Отмена - False."""
        t0 = time.time()
        while not self.try_acquire():
            if cancel is not None and cancel.is_set():
                return False
            if time.time() - t0 > wait:
                return False
            if on_wait:
                on_wait(owner(self.path))
            time.sleep(2)
        return True

    def release(self) -> None:
        with self._mu:
            if not self.held:
                return
            self.held = False
            if held_outside():
                return
            if owner_pid(self.path) == os.getpid():
                shutil.rmtree(self.path, ignore_errors=True)


__all__ = ["HeavyLock", "lock_path", "owner", "ours", "held_outside", "OWNER"]
