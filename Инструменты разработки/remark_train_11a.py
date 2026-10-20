#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, Д: тексты набора для дообучения - заново ударениями части А (по
предложению, согласование, словари). Набор Герасимова готовился 20.10 ночью ещё
разметкой 3.9. Меняет только поле "text" (U+0301 после ударной гласной).

    python "Инструменты разработки/remark_train_11a.py" "Вячеслав Герасимов"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json


def main(voice: str) -> int:
    from core.stress import add_stress_text, to_combining
    p = _os.path.expanduser(f"~/.cache/text2audio/train/{voice}/train_raw.jsonl")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    changed = 0
    for r in rows:
        new = to_combining(add_stress_text(r["text_plain"]))
        changed += new != r["text"]
        r["text"] = new
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{voice}: фраз {len(rows)}, разметка изменилась у {changed}")
    return 0


if __name__ == "__main__":
    _sys.exit(main(_sys.argv[1]))
