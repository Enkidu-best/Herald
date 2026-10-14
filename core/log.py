"""Журнал проблем: ~/Library/Logs/Herald/herald.log.

Раньше ошибки в десятках мест молча глотались (`except Exception: pass`), и
сбой выглядел как «программа сделала что-то не то». Так, например, месяцами
терялись ударения во всех фразах с именами. Теперь всё, что меняет результат,
пишется сюда с подробностями, а коротко - в окно (через stderr, который окно
показывает).
"""

from __future__ import annotations

import logging
import os
import sys
import traceback

LOG_DIR = os.path.expanduser("~/Library/Logs/Herald")
LOG_PATH = os.path.join(LOG_DIR, "herald.log")

_logger: logging.Logger | None = None


def _get() -> logging.Logger:
    global _logger
    if _logger is None:
        from logging.handlers import RotatingFileHandler
        lg = logging.getLogger("herald")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            h = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=3,
                                    encoding="utf-8")
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            lg.addHandler(h)
        except OSError:
            lg.addHandler(logging.NullHandler())
        _logger = lg
    return _logger


def log_problem(what: str, exc: BaseException | None = None, *, show: bool = True) -> None:
    """Записать проблему в журнал; show=True - ещё и одной строкой в окно."""
    detail = ""
    if exc is not None:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _get().warning("%s%s", what, ("\n" + detail) if detail else "")
    if show:
        short = f": {type(exc).__name__}: {exc}" if exc is not None else ""
        print(f"⚠︎ {what}{short}", file=sys.stderr)


def log_info(what: str) -> None:
    _get().info("%s", what)
