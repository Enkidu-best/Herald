#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, п. 6: 300 слов из книг - по звуку, до и после части А.

Фраза с целевым словом размечается так, как её размечает программа внутри абзаца
(«до» - как в 3.9, главой одним куском; «после» - настройки по умолчанию), по 3 фразы
в кусок, синтез Qwen ru-stress с настройками по умолчанию голосом «Диктор 1»
(образец банка «Голос 06»). Верно ли - meter2 против ВЕРНОГО ударения целевого слова.
«После» синтезируется на всех словах; «до» - только там, где разметка «до» и
«после» различается (на остальных звук тот же в ожидании - разница в звуке
даёт только разница в разметке). Под ../.heavy-lock.

    python "Инструменты разработки/stress_audio300_11a.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import importlib.util
import json
import re

OUT = "Архив/ударения 11а"
VOICE = _os.path.abspath("Архив/голоса диктора 3.9/образцы/Голос 06.wav")


def _steps():
    spec = importlib.util.spec_from_file_location("st11a", "Инструменты разработки/stress_steps_11a.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sentence_marked(marked: str, para: str, start: int, end: int) -> str:
    """Кусок размеченного абзаца, соответствующий para[start:end]."""
    pm, i, j = {}, 0, 0
    while i <= len(para) and j <= len(marked):
        if j < len(marked) and marked[j] == "+":
            j += 1
            continue
        pm[i] = j
        i += 1
        j += 1
    a = pm.get(start, 0)
    if a and marked[a - 1] == "+":
        a -= 1
    return marked[a:pm.get(end, len(marked))]


def mark(items, S, P, step_mod, step: str) -> list[str]:
    step_mod.configure(step, S, None)
    cache, out = {}, []
    for it in items:
        para = it["абзац"]
        if para not in cache:
            cache[para] = P._prepare_text(para, "+", True, True, "ru")
        a = para.index(it["фраза"])
        s = sentence_marked(cache[para], para, a, a + len(it["фраза"]))
        if s.replace("+", "") != it["фраза"]:              # буквы разошлись - фраза отдельно
            s = P._prepare_text(it["фраза"], "+", True, True, "ru")
        out.append(s)
    return out


def truth_text(it) -> str:
    w, t = it["слово"], it["верно"]
    return it["фраза"][:it["позиция"]] + t + it["фраза"][it["позиция"] + len(w):]


def run(items, idx, marked, eng, meter, P, label):
    import soundfile as sf
    groups, cur = [], []
    for i in idx:
        if len(cur) == 3 or any(items[i]["слово"].lower() == items[j]["слово"].lower() for j in cur):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    bad_all, n = [], 0
    d = _os.path.join(OUT, f"звук 300 {label}")
    _os.makedirs(d, exist_ok=True)
    for gi, g in enumerate(groups):
        text = P._with_end_punct(" ".join(P._with_end_punct(marked[i]) for i in g))
        au = eng.synth_chunk(text, VOICE)
        sf.write(_os.path.join(d, f"{gi + 1:03d}.wav"), au, eng.sample_rate)
        truth = " ".join(P._with_end_punct(truth_text(items[i])) for i in g)
        words = {items[i]["слово"].lower() for i in g}
        bad = {b.replace("+", "").lower() for b in meter.wrong_words(au, eng.sample_rate, truth, only=words,
                                                                      min_margin=0.0)}
        n += len(words)
        bad_all += sorted(bad)
        print(label, gi + 1, len(groups), sorted(bad), flush=True)
    return n, bad_all


def main() -> int:
    import core.pipeline as P
    from core import stress as S, meter
    from core.engines.mlxaudio import MlxAudioEngine
    from core.profiles import PROFILES, engine_kwargs
    stm = _steps()
    items = [it for it in json.load(open(f"{OUT}/книги_300.json", encoding="utf-8")) if it.get("верно")]
    before = mark(items, S, P, stm, "0")
    after = mark(items, S, P, stm, "итог")
    changed = [i for i in range(len(items)) if before[i] != after[i]]
    print(f"слов: {len(items)}, разметка различается: {len(changed)}", flush=True)
    eng = MlxAudioEngine(**engine_kwargs(PROFILES["qwen"]))
    eng.ensure_loaded()
    n_a, bad_a = run(items, list(range(len(items))), after, eng, meter, P, "после")
    n_b, bad_b = run(items, changed, before, eng, meter, P, "до (где разметка другая)")
    eng.close()
    changed_words = {items[i]["слово"].lower() for i in changed}
    bad_a_changed = [w for w in bad_a if w in changed_words]
    res = {"слов": n_a, "после: не так по звуку": len(bad_a), "после: слова": bad_a,
           "разметка различается": len(changed),
           "до, на различающихся: не так": len(bad_b), "после, на них же: не так": len(bad_a_changed),
           "до: всего не так (оценка)": len(bad_a) - len(bad_a_changed) + len(bad_b), "до: слова": bad_b}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    json.dump(res, open(f"{OUT}/звук 300.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
