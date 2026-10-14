"""Не давать Mac уснуть, пока считается книга.

Книга считается часами (6,5 ч у «Жажды власти»), а macOS по умолчанию усыпляет
простаивающую систему: синтез вставал посреди ночи. На время работы держим
`caffeinate -i -s -w <pid>`:
  -i  не засыпать от простоя;
  -s  не засыпать системе, пока Mac на питании от сети;
  -w  caffeinate сам завершится, если наш процесс упадёт, - Mac не останется
      навсегда бодрствующим из-за сбоя.
На «Паузе» запрет снимаем: пауза бывает «на ночь», и держать Mac без дела
незачем. По «Стопу» и по окончании снимаем всегда.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import threading

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_wanted = 0          # сколько участков кода сейчас просят не спать


def _start() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return
    exe = shutil.which("caffeinate")
    if not exe:
        return
    try:
        _proc = subprocess.Popen([exe, "-i", "-s", "-w", str(os.getpid())],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        from .log import log_problem
        log_problem("не удалось запретить сон Mac на время озвучки", e)
        _proc = None


def _stop() -> None:
    global _proc
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=3)
        except Exception:
            pass                    # процесс уже завершился — это и нужно
        _proc = None


def active() -> bool:
    return _proc is not None and _proc.poll() is None


@contextlib.contextmanager
def keep_awake():
    """Пока блок выполняется, Mac не уснёт. Вложенные блоки допустимы."""
    global _wanted
    with _lock:
        _wanted += 1
        _start()
    try:
        yield
    finally:
        with _lock:
            _wanted -= 1
            if _wanted <= 0:
                _wanted = 0
                _stop()


def pause() -> None:
    """«Пауза»: Mac можно усыплять."""
    with _lock:
        _stop()


def resume() -> None:
    """«Продолжить»: снова не спать, если работа ещё идёт."""
    with _lock:
        if _wanted > 0:
            _start()
