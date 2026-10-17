#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Где теряется ударение (3.7, п. 1): по шагам подготовки текста.

На каждую фразу (файл: «фраза | слово | верное ударение») - ударение в целевом
слове после RUAccent, после silero-stress, после их связки (как в программе) и
после всей подготовки (числа, ударения, слова пользователя) - и верно ли.

    python "Инструменты разработки/bench_stress_stages.py" ударения_30.txt омографы_50.txt
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import re

THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 0.95, 1.01)


def main() -> int:
    import core.pipeline as P
    from core import stress as ST
    files = [a if _os.path.isabs(a) else _os.path.join("Инструменты разработки", a)
             for a in _sys.argv[1:]]
    for f in files:
        rows = []
        for line in open(f, encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            phrase, word, right = [x.strip() for x in line.split("|")]
            def pick(x):
                for m in re.finditer(r"[А-Яа-яЁё+]+", x):
                    if m.group().replace("+", "").lower() == word.lower():
                        w = m.group().lower()
                        # «ё» всегда ударна: «всё» и «вс+ё» - одно и то же
                        return w.replace("+ё", "ё") if "ё" in w else w
                return "?"
            def ok(w):
                r = right.lower().replace("+ё", "ё")
                if sum(c in "аеёиоуыэюя" for c in r.replace("+", "")) == 1:
                    return w.replace("+", "") == r.replace("+", "")   # односложное
                return w == r
            # пороги: silero - только где RUAccent не уверен
            by_thr = {}
            for thr in THRESHOLDS:
                ST.SILERO_BELOW = thr
                by_thr[thr] = pick(ST.add_stress_text(phrase))
            ST.SILERO_BELOW = 0.0
            st = {"RUAccent": pick(ST.add_stress(phrase)),
                  "silero": pick(ST._silero_text(phrase)),
                  "вместе": pick(ST.add_stress_text(phrase)),
                  "подготовка": pick(P._prepare_text(phrase, "+", True, True, "ru"))}
            st.update({f"silero ниже {t}": v for t, v in by_thr.items()})
            rows.append((phrase, word, right, st, {k: ok(v) for k, v in st.items()}))
        n = len(rows)
        print(f"\n== {_os.path.basename(f)}: {n} фраз")
        for k in ["RUAccent", "silero", "вместе", "подготовка"] + [f"silero ниже {t}" for t in THRESHOLDS]:
            print(f"   {k:11s}: верно {sum(r[4][k] for r in rows)} из {n}")
        for phrase, word, right, st, ok in rows:
            if not all(ok.values()):
                print(f"   {right:12s} RUAccent {st['RUAccent']:12s} silero {st['silero']:12s} "
                      f"вместе {st['вместе']:12s} | {phrase[:55]}")
    _sys.stdout.flush()
    _os._exit(0)


if __name__ == "__main__":
    main()
