#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Паузы: после предложения, после абзаца, между репликами (доработки 3.0, п. 2).

Быстрая часть (без модели):
  * нарезка диалогового фрагмента: какая пауза где стоит - и между кусками,
    и отметками внутри склеенного куска;
  * правка паузы внутри тишины (_set_pause) и склейка кусков (_join) на
    искусственном сигнале: пауза выходит заданной длины, звук не задет.
Полная часть (с моделью): тот же диалог озвучивается целиком, и по меткам
времени слов (Whisper large-v3-turbo, независимый от программы) измеряются
настоящие паузы на стыках. Между репликами должно выйти заметно длиннее, чем
между предложениями, и каждая пауза - близко к заданной.

    python "Инструменты разработки/test_pauses.py" [--fast]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import os
import contextlib
import io
import tempfile

import numpy as np

SR = 24000
GAP, PARA, DIALOG = 0.4, 0.8, 1.1
DIALOG_TEXT = """Он вошёл в комнату и огляделся. Было темно и тихо.

— Кто здесь? — спросил он.

— Я, — ответил голос из угла. — Не бойся.

— Покажись немедленно!

— Не могу.

Тишина. Потом что-то зашуршало у окна."""
fails: list[str] = []


def must(ok: bool, msg: str):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        fails.append(msg)


def tone(sec: float, f: float = 220.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence_runs(a: np.ndarray) -> list[tuple[float, float]]:
    from core.pipeline import _silent_frames, SIL_FRAME
    sil, _w = _silent_frames(a, SR)
    runs, cur = [], None
    for i, s in enumerate(sil):
        if s:
            cur = (cur[0], i + 1) if cur else (i, i + 1)
        elif cur:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return [(s * SIL_FRAME, e * SIL_FRAME) for s, e in runs]


def fast_part():
    from core.pipeline import _chunk_paragraphs, _set_pause, _join, CLAUSE_GAP
    print("Нарезка диалога:")
    ch = _chunk_paragraphs(DIALOG_TEXT, 280, GAP, PARA, DIALOG)
    flat = []                                # (текст до стыка, пауза на стыке)
    for text, gap, _mul, marks in ch:
        last = 0
        for pos, g in marks:
            flat.append((text[last:pos].strip(), g))
            last = pos
        flat.append((text[last:].strip(), gap))
    for t, g in flat:
        print(f"    {g:.2f} с  после «{t[-40:]}»")
    want = {"Было темно и тихо.": PARA, "— спросил он.": DIALOG, "голос из угла.": GAP,
            "— Не бойся.": DIALOG, "Покажись немедленно!": DIALOG, "— Не могу.": PARA,
            "и огляделся.": GAP}
    for end, g in want.items():
        got = [p for t, p in flat if t.endswith(end)]
        must(got == [g], f"после «{end}» пауза {g} (получено {got})")
    must(all(g in (GAP, PARA, DIALOG, CLAUSE_GAP) for _t, g in flat), "только заданные паузы")
    must(sum(1 for _t, g in flat if g == DIALOG) == 3, "три стыка реплика-реплика")

    print("Правка паузы внутри тишины:")
    a = np.concatenate([tone(0.5), np.zeros(int(0.6 * SR), np.float32), tone(0.5, 330)])
    for target in (1.1, 0.35, 0.8, 0.2):          # раздвинуть и сжать
        b = _set_pause(a, SR, int(0.5 * SR), int(1.1 * SR), target)
        runs = [r for r in silence_runs(b) if r[1] - r[0] > 0.05]
        got = runs[0][1] - runs[0][0] if runs else 0
        must(abs(got - target) <= 0.03, f"0,60 -> {target:.2f} с: вышло {got:.2f} с")
        loud = np.abs(b) > 0.05
        must(abs(loud.sum() / SR - np.sum(np.abs(a) > 0.05) / SR) < 0.02,
             "звук (тон) не задет")
    print("Склейка кусков:")
    p1 = np.concatenate([tone(0.6), np.zeros(int(0.25 * SR), np.float32)])
    p2 = np.concatenate([np.zeros(int(0.05 * SR), np.float32), tone(0.6, 330)])
    j = _join([(p1, DIALOG), (p2, PARA)], SR)
    runs = [r for r in silence_runs(j) if r[1] - r[0] > 0.05]
    got = runs[0][1] - runs[0][0] if runs else 0
    must(abs(got - DIALOG) <= 0.03, f"пауза между кусками {DIALOG} с: вышло {got:.2f} с")
    tail = len(j) / SR - (runs[-1][0] if runs else 0)
    must(abs(tail - PARA) <= 0.03, f"после последнего куска {PARA} с: вышло {tail:.2f} с")


def word_times(audio: np.ndarray) -> list[tuple[str, float, float]]:
    import mlx_whisper
    import soundfile as sf
    from core.pipeline import _words
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SR)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = mlx_whisper.transcribe(f.name, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                       language="ru", word_timestamps=True, verbose=False,
                                       condition_on_previous_text=False)
    finally:
        os.unlink(f.name)
    return [(ww, w["start"], w["end"]) for s in r["segments"] for w in s["words"]
            for ww in _words(w["word"])]


def full_part():
    import mlx.core as mx
    from core.engines import get_engine
    from core.pipeline import _prepare_text, _chunk_paragraphs, _synth_chunks
    print("\nОзвучка диалога и замер настоящих пауз:")
    from core import profiles
    from core.engines import clone_engine
    eng = get_engine(clone_engine(), language="ru", **profiles.get(None).kwargs)
    eng.ensure_loaded()
    text = _prepare_text(DIALOG_TEXT, "+", True, True, "ru")
    ch = _chunk_paragraphs(text, eng.max_chunk_chars, GAP, PARA, DIALOG)
    mx.random.seed(5)
    st: dict = {}
    audio = _synth_chunks(eng, "Вячеслав Герасимов", ch, SR, GAP, use_cache=False, stats=st)
    print(f"    кусков {len(ch)}, пауз внутри кусков доведено: {st['pauses_fit']}")
    wt = word_times(audio)
    runs = silence_runs(audio)
    # стыки: (последнее слово перед, первое слово после, ожидаемая пауза)
    joints = [("огляделся", "было", GAP), ("тихо", "кто", PARA), ("он", "я", DIALOG),
              ("угла", "не", GAP), ("бойся", "покажись", DIALOG),
              ("немедленно", "не", DIALOG), ("могу", "тишина", PARA)]
    seen = {"sent": [], "dialog": []}
    skipped: list[str] = []
    long_miss = 0
    i = 0
    def same(w, want):                   # распознаватель вправе спутать окончание
        return w[:4] == want[:4]
    for a_w, b_w, g in joints:
        while i + 1 < len(wt) and not (same(wt[i][0], a_w) and same(wt[i + 1][0], b_w)):
            i += 1
        if i + 1 >= len(wt):
            # модель пропустила слово у стыка - это ошибка чтения, не паузы
            print(f"  - стык «{a_w} | {b_w}» пропущен: модель не произнесла слово "
                  f"(услышано: {' '.join(w for w, _s, _e in wt[:14])}…)")
            skipped.append(a_w)
            i = 0
            continue
        # метки начала слова у Whisper сдвинуты на десятые доли секунды, поэтому
        # берём паузу, которая НАЧИНАЕТСЯ между словами, целиком
        lo, hi = wt[i][2] - 0.25, wt[i + 1][1] + 0.1
        inside = [e - s for s, e in runs if lo <= s <= hi]
        got = max(inside) if inside else 0.0
        if g == GAP and got < 0.15:
            # между предложениями модель паузы не дала - программа её не трогает
            # (растягивать короткую тишину опасно: это может быть смычка согласной)
            print(f"  - «{a_w} | {b_w}»: модель не дала паузы ({got:.2f} с), оставлено как есть")
        elif g == GAP:
            must(abs(got - g) <= 0.2, f"«{a_w} | {b_w}»: пауза {got:.2f} с (задано {g})")
        else:
            ok = abs(got - g) <= 0.2
            print(("  ✓ " if ok else "  ~ ") + f"«{a_w} | {b_w}»: пауза {got:.2f} с (задано {g})")
            long_miss += 0 if ok else 1
        if g in (DIALOG, GAP):
            seen["dialog" if g == DIALOG else "sent"].append(got)
        i += 1
    must(len(skipped) <= 1, f"пропущенных стыков не больше одного ({skipped})")
    # Паузу внутри куска программа ставит по меткам слов; если распознаватель
    # не сопоставил слова у стыка, пауза остаётся модельной. Одна такая на
    # фрагмент допустима (замер: 17 из 18 длинных пауз точно в цель).
    must(long_miss <= 1, f"длинных пауз мимо цели не больше одной ({long_miss})")
    if seen["dialog"] and seen["sent"]:
        d, s = np.mean(seen["dialog"]), np.mean(seen["sent"])
        must(d >= s + 0.4, f"между репликами заметно длиннее: {d:.2f} против {s:.2f} с")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="без модели")
    a = ap.parse_args()
    fast_part()
    if not a.fast:
        full_part()
    if fails:
        print(f"\nПРОВАЛ: {len(fails)}")
        return 1
    print("\nOK: паузы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
