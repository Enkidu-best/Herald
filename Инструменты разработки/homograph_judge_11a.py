#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, п. 2: кто вернее решает омограф в контексте.

Набор: омографы_50 (25 слов × 2 смысла), омографы из ударения_50_39, омографы
из 300 слов книг (с соседними предложениями), три ошибки длинного куска 3.9
(«Связи требует…», «к Малому Доминиону», «у Малого Доминиона»).
Судьи: RUAccent (по предложению), silero-stress, как сейчас (silero, если RUAccent
не уверен), языковая модель (Ollama) - на всех и только на сомнительных
(RUAccent не уверен или разошёлся с silero). Под ../.heavy-lock.

    python "Инструменты разработки/homograph_judge_11a.py" [qwen3:14b qwen3.5:9b]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import json
import re
import time

T = "Инструменты разработки"
OUT = "Архив/ударения 11а"
WORD = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")
LONG = [  # длинный кусок 3.9: ошибки, которые слышал Андрей (до, фраза, после, слово, верно)
    ("И игнорировал подрагивающий в углу экрана индикатор срочного вызова.",
     "Связи требует канцелярия Владыки Императорского Доминиона, желают получить представление о "
     "положении дел.", "Подождут.", "Связи", "св+язи"),
    ("Допустим, что это было сделано в качестве жеста доброй воли.",
     "Это было сделано по отношению к Малому Доминиону Чиалори, к которому принадлежат владельцы "
     "добывающей компании.", "", "Малому", "м+алому"),
    ("Поэтому сенаторам от разных регионов выгодно объединяться в блоки.",
     "Больше всего сенаторов у Малого Доминиона Чиалори.",
     "Конечно, их численность почти вдвое уступает количеству сенаторов от Большой Тройки.",
     "Малого", "м+алого"),
]


def rows_file(name):
    for line in open(f"{T}/{name}", encoding="utf-8"):
        if line.strip() and not line.startswith("#"):
            p, w, t = [x.strip() for x in line.split("|")]
            yield ("", p, "", w, t)


def cases():
    from core import stress as S, stressdict as D
    acc = S._get_accentizer()
    homo = lambda w: S._key(w) in acc.omographs or S._key(w) in getattr(acc, "yo_homographs", {}) \
        or len(D.options(w)) > 1  # noqa: E731
    out = []
    for r in list(rows_file("омографы_50.txt")) + list(rows_file("ударения_50_39.txt")):
        if homo(r[3]) and r not in out:
            out.append(r)
    for it in json.load(open(f"{OUT}/книги_300.json", encoding="utf-8")):
        if it.get("верно") and homo(it["слово"]):
            # второй вариант должен быть настоящим словом, а не опечаткой словаря
            out.append((it["до"], it["фраза"], it["после"], it["слово"], it["верно"]))
    out += LONG
    return out


def pick(marked: str, word: str) -> str:
    for m in WORD.finditer(marked):
        if S_key(m.group()) == S_key(word):
            return m.group().lower()
    return "?"


def S_key(w):
    return w.replace("+", "").lower().replace("ё", "е")


def same(a: str, truth: str) -> bool:
    from core.stressdict import stress_of
    a, truth = a.lower(), truth.lower()
    if sum(c in "аеёиоуыэюя" for c in S_key(truth)) == 1:          # все / всё
        return a.replace("+", "") == truth.replace("+", "")
    return stress_of(a) == stress_of(truth)


def variants_of(word: str):
    """Варианты ударения (с «+») и грамматика каждого: справочник, иначе RUAccent."""
    from core import stress as S, stressdict as D
    acc = S._get_accentizer()
    k = S._key(word)
    v = D.variants(word)
    vs = sorted(x for x in v if any(c == "+" for c in x) or "ё" in x)
    if "ё" in "".join(vs) and k in v:           # все / всё: форма без «ё» тоже вариант
        vs = sorted(set(vs) | {k})
    if len(vs) < 2:
        vs = sorted(set(acc.omographs.get(k, [])) | set(getattr(acc, "yo_homographs", {}).get(k, []) and
                                                        [k, acc.yo_homographs[k]] or []))
    return vs, [v.get(x, set()) for x in vs]


def main() -> int:
    from core import stress as S, homograph_llm as L
    models = _sys.argv[1:] or ["qwen3:14b"]
    acc = S._get_accentizer()
    S._wrap_classify(acc)
    cs = cases()
    print(f"омографов в наборе: {len(cs)}", flush=True)
    res = {k: 0 for k in ("RUAccent", "silero", "как сейчас")}
    detail = []
    for before, s, after, w, t in cs:
        S._CONF.clear()
        ru = pick(S._restore_case(s, acc.process_all(S._lower_titles(s))), w)
        conf = S._CONF.get(S._key(w), 1.0)
        si = pick(S._silero_text(s), w)
        now = si if (conf < S.SILERO_BELOW and ru != si) else ru
        vs, gr = variants_of(w)
        detail.append({"до": before, "фраза": s, "после": after, "слово": w, "верно": t, "RUAccent": ru,
                       "уверенность": round(conf, 2), "silero": si, "как сейчас": now, "варианты": vs,
                       "_гр": gr, "сомнение": conf < 0.9 or not same(ru, si)})
        for k, v in (("RUAccent", ru), ("silero", si), ("как сейчас", now)):
            res[k] += same(v, t)
    print("сомнительных:", sum(d["сомнение"] for d in detail), flush=True)
    for m in models:                       # по одной модели: две в память не помещаются
        t0 = time.time()
        for d in detail:
            vs = d["варианты"]
            n = L.choose_in_context(d["слово"], d["фраза"], vs, d["до"], d["после"], d["_гр"], model=m) \
                if len(vs) > 1 else None
            d[m] = vs[n] if n is not None else d["как сейчас"]
            d[m + " на сомнительных"] = d[m] if d["сомнение"] else d["RUAccent"]
            for k in (m, m + " на сомнительных"):
                res[k] = res.get(k, 0) + same(d[k], d["верно"])
        res[m + ", с на слово"] = round((time.time() - t0) / len(detail), 2)
        print(m, {k: v for k, v in res.items() if k.startswith(m)}, flush=True)
        L.unload(m)
    for d in detail:
        d.pop("_гр")
    print(f"\nиз {len(cs)}:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    json.dump({"итог": res, "из": len(cs), "по словам": detail},
              open(f"{OUT}/судья омографов.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
