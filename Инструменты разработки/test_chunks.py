#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка кусков: ни одно слово не разрезано между кусками (11б: «капит+а. / на.»).

Берёт размеченные тексты, которые конвейер уже сохранял («_служебное/текст с ударениями.txt»
в Архив/ и Сравнение*/), режет их на куски, как для F5 (187 знаков) и Qwen (600), и сверяет
слова: склейка кусков должна дать те же слова, что и текст, - без «капит+а» + «на».
Сравнение со старым кодом (git HEAD~) - сколько слов он резал. Моделей не нужно.

    python "Инструменты разработки/test_chunks.py"
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])

import glob
import re
import subprocess
import types


def words(t: str) -> list[str]:
    return re.findall(r"[А-Яа-яЁёA-Za-z+]+(?:-[А-Яа-яЁёA-Za-z+]+)*", t)


def broken(P, text: str, max_chars: int) -> list[str]:
    """Слова текста, которых нет в склейке кусков (разрезанные)."""
    out = []
    for para in [p for p in text.split("\n") if p.strip() and not p.startswith("===")]:
        chunks = P._chunk_paragraphs(para, max_chars, 0.4, 0.8, 1.1)
        got = [w for c in chunks for w in words(c[0])]
        src = words(para)
        if got != src:
            import difflib
            sm = difflib.SequenceMatcher(None, src, got, autojunk=False)
            out += [" ".join(src[i1:i2]) + " -> " + " | ".join(got[j1:j2])
                    for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    return out


def old_pipeline():
    """core/pipeline.py до исправления 11б - только функции резки."""
    src = subprocess.run(["git", "show", "f669a06:core/pipeline.py"], capture_output=True, text=True).stdout
    m = types.ModuleType("old_pipeline")
    m.__dict__["__name__"] = "core.old_pipeline"
    m.__dict__["__package__"] = "core"
    sys.modules["core.old_pipeline"] = m           # dataclass ищет свой модуль здесь
    exec(compile(src, "old_pipeline.py", "exec"), m.__dict__)
    return m


def main() -> int:
    import core.pipeline as P
    files = sorted(set(glob.glob("Архив/**/_служебное/текст с ударениями.txt", recursive=True)))
    texts = {}
    for f in files:
        t = open(f, encoding="utf-8").read()
        texts.setdefault(t, f)
    print(f"текстов: {len(texts)} (из {len(files)} файлов)")
    old = old_pipeline()
    fails = 0
    for mc in (187, 600):
        n_old = n_new = 0
        for t, f in texts.items():
            b_old, b_new = broken(old, t, mc), broken(P, t, mc)
            n_old += len(b_old)
            n_new += len(b_new)
            for b in b_new[:3]:
                print(f"  ПЛОХО {mc}: {b}   ({f})")
        print(f"куски до {mc}: разрезанных слов было {n_old}, стало {n_new}")
        fails += n_new
    print("всё в порядке" if not fails else f"ПРОВАЛЕНО: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
