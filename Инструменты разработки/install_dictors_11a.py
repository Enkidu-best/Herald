#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Б: голоса «Диктор 1-3» - в список голосов движка Qwen ru-stress.

Голоса 01, 02, 03 папки «Сравнение 3.9 - голоса диктора на выбор» - образцы из
записей самой ru-stress (банк 3.9: Голос 06, 05, 03). Для 01 и 02 - «живой»
образец, если он выиграл сравнение (lively_sample_11a.py: размах тона больше без
роста WER), иначе обычный. Кладутся в ~/.cache/text2audio/voices/qwen/ - их видит
только движок Qwen. Только для личного использования: в git и в установщик не
входят (происхождение голосов ru-stress не выяснено, см. docs/Приёмка 3.8.md).

    python "Инструменты разработки/install_dictors_11a.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import shutil

BANK = "Архив/голоса диктора 3.9"
DICTORS = [("Диктор 1", "01", "Голос 06"), ("Диктор 2", "02", "Голос 05"), ("Диктор 3", "03", "Голос 03")]


def lively_wins(num: str) -> bool:
    p = f"{BANK}/живой образец/итог.json"
    if not _os.path.exists(p):
        return False
    r = json.load(open(p, encoding="utf-8")).get(num, {}).get("сравнение", {})
    a, b = r.get("обычный образец"), r.get("живой образец")
    return bool(a and b and b["размах тона"] > a["размах тона"] and b["WER"] <= a["WER"] + 0.01)


def main() -> int:
    from core.engines.mlxaudio import own_voices_dir
    d = own_voices_dir()
    _os.makedirs(d, exist_ok=True)
    for name, num, bank in DICTORS:
        src = f"{BANK}/живой образец/Голос {num} - живой" if lively_wins(num) else f"{BANK}/образцы/{bank}"
        for ext in (".wav", ".txt"):
            shutil.copy(src + ext, _os.path.join(d, name + ext))
        print(f"{name} (папка 3.9: Голос {num}) <- {src}.wav")
    # 21.10: Андрею очень понравились оба «живых» образца (01 и 02) - готовы для записи;
    # «Диктор 1» остаётся прежним, «живой» 01 - отдельным голосом рядом
    src = f"{BANK}/живой образец/Голос 01 - живой"
    for ext in (".wav", ".txt"):
        shutil.copy(src + ext, _os.path.join(d, "Диктор 1 (живой)" + ext))
    print(f"Диктор 1 (живой) <- {src}.wav")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
