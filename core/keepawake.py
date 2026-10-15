"""Не давать компьютеру уснуть, пока считается книга.

Книга считается часами (6,5 ч у «Жажды власти»), а система по умолчанию
усыпляет простаивающий компьютер: синтез вставал посреди ночи. На время работы:
  * macOS - `caffeinate -i -s -w <pid>`: не засыпать от простоя и на питании от
    сети; -w - caffeinate сам завершится, если наш процесс упадёт;
  * Windows - SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) из
    отдельного потока (флаг действует, пока жив поток, который его поставил);
  * Linux - `systemd-inhibit --what=sleep:idle` вокруг `sleep infinity`.
На «Паузе» запрет снимаем: пауза бывает «на ночь», и держать компьютер без
дела незачем. По «Стопу» и по окончании снимаем всегда.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import threading

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_win_stop: threading.Event | None = None
_wanted = 0          # сколько участков кода сейчас просят не спать


def _command() -> list[str] | None:
    if sys.platform == "darwin":
        exe = shutil.which("caffeinate")
        return [exe, "-i", "-s", "-w", str(os.getpid())] if exe else None
    exe = shutil.which("systemd-inhibit")
    if exe:
        return [exe, "--what=sleep:idle", "--who=Herald", "--why=Озвучка книги",
                "--mode=block", "sleep", "infinity"]
    return None


def _win_hold(stop: threading.Event) -> None:
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    k = ctypes.windll.kernel32
    k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        stop.wait()
    finally:
        k.SetThreadExecutionState(ES_CONTINUOUS)


def _start() -> None:
    global _proc, _win_stop
    if sys.platform.startswith("win"):
        if _win_stop is None:
            _win_stop = threading.Event()
            threading.Thread(target=_win_hold, args=(_win_stop,), daemon=True,
                             name="keepawake").start()
        return
    if _proc is not None and _proc.poll() is None:
        return
    cmd = _command()
    if not cmd:
        return
    try:
        _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        from .log import log_problem
        log_problem("не удалось запретить сон компьютера на время озвучки", e)
        _proc = None


def _stop() -> None:
    global _proc, _win_stop
    if _win_stop is not None:
        _win_stop.set()
        _win_stop = None
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=3)
        except Exception:
            pass                    # процесс уже завершился — это и нужно
        _proc = None


def active() -> bool:
    if _win_stop is not None:
        return True
    return _proc is not None and _proc.poll() is None


@contextlib.contextmanager
def keep_awake():
    """Пока блок выполняется, компьютер не уснёт. Вложенные блоки допустимы."""
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
    """«Пауза»: компьютер можно усыплять."""
    with _lock:
        _stop()


def resume() -> None:
    """«Продолжить»: снова не спать, если работа ещё идёт."""
    with _lock:
        if _wanted > 0:
            _start()
