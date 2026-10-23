#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Статическая проверка кода программы (стандарт 6.11, задание 13 А).

    ruff check --select F821,F841,B023  tts_gui.py cli.py player.py theme.py core/

F821 - неопределённое имя (так падало сообщение об ошибке: `except Exception as e:
self.ui(lambda: ...e...)` - Python удаляет `e` после блока); F841 - лишняя переменная;
B023 - замыкание на переменную цикла. Все три - 0. Ruff: из окружения (`pip install
ruff`), иначе через `uvx`; нет ни того, ни другого - ошибка с подсказкой.

    python "Инструменты разработки/test_lint.py"
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["tts_gui.py", "cli.py", "player.py", "theme.py", "core"]
RULES = "F821,F841,B023"
RUFF = "0.6.9"


def ruff_cmd() -> list[str] | None:
    try:
        import ruff  # noqa: F401
        return [sys.executable, "-m", "ruff"]
    except ImportError:
        pass
    if shutil.which("ruff"):
        return ["ruff"]
    if shutil.which("uvx"):
        return ["uvx", f"ruff@{RUFF}"]
    return None


def main() -> int:
    cmd = ruff_cmd()
    if cmd is None:
        print("  ПЛОХО  ruff не найден: pip install ruff (или uv)")
        return 1
    r = subprocess.run(cmd + ["check", "--select", RULES, "--no-cache", "--output-format", "concise"]
                       + FILES, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    out = (r.stdout + r.stderr).strip()
    print(out)
    ok = r.returncode == 0
    print(f"  {'OK  ' if ok else 'ПЛОХО'}  ruff {RULES}: {'0 замечаний' if ok else 'есть замечания'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
