#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка «это точно нужные слова»: распознаём готовый wav и сверяем с текстом.

Нужна потому, что на слух каждый вариант не переслушать, а «речеподобная каша»
по громкости и длительности выглядит как нормальная речь. Сравниваем по словам.

    python verify.py "Книга/01 Глава первая.mp3" --book "Книга.txt" --index 1

Отдельно проверяет ПЕРВОЕ и ПОСЛЕДНЕЕ слово: именно их модель склонна глотать,
а в общем проценте одно слово почти не видно.
"""
from __future__ import annotations

import os as _os, sys as _sys
# скрипт лежит в подпапке, а core/ — в корне проекта
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])
import os, re, sys, difflib

os.environ.setdefault("PYTHONHASHSEED", "0")

from core.normalize import normalize_text, sentenize
from core.stress import add_stress

TEXT_FILE = "Пример — проба пера.txt"


def words(s: str) -> list[str]:
    s = s.lower().replace("+", "").replace("ё", "е")
    return re.sub(r"[^а-яa-z ]", " ", s).split()


def expected(book: str, index: int) -> list[str]:
    """Слова, которые ДОЛЖНЫ прозвучать в файле №index этой книги.

    Берём ровно тот текст, который пайплайн отдал бы на синтез, иначе сравнение
    краёв врёт: раньше здесь склеивались первые N кусков по 200 символов, и
    «первое слово» эталона не совпадало с первым словом файла просто из-за
    другой нарезки.
    """
    from core.readers import read_document
    from core.chapters import build_chapters
    doc = read_document(book)
    chs = build_chapters(doc)
    if not 1 <= index <= len(chs):
        raise SystemExit(f"в книге {len(chs)} файлов, запрошен {index}")
    return words(normalize_text(chs[index - 1].text, expand_numbers=True))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--book", default=TEXT_FILE, help="книга, из которой делали файлы")
    ap.add_argument("--index", type=int, default=1, help="номер файла в книге (с 1)")
    args = ap.parse_args()
    files = args.files
    if not files:
        print(__doc__); return 1
    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8")
    want = expected(args.book, args.index)
    print(f"ожидаем {len(want)} слов\n")
    rc = 0
    for f in files:
        segs, _ = m.transcribe(f, language="ru")
        got = words("".join(x.text for x in segs))
        sm = difflib.SequenceMatcher(None, want, got)
        hit = sum(b.size for b in sm.get_matching_blocks())
        pct = 100.0 * hit / max(1, len(want))
        mark = "OK  " if pct >= 80 else "ПЛОХО"
        print(f"{mark} {pct:5.1f}% слов  {os.path.basename(f)}  ({len(got)} распознано)")
        # края проверяем отдельно: именно первое и последнее слово модель
        # склонна глотать, а в общем проценте одно слово почти не видно
        if got and want:
            if got[0] != want[0]:
                print(f"      ⚠︎ ПЕРВОЕ слово: ждали «{want[0]}», слышно «{got[0]}»")
                rc = 1
            if got[-1] != want[-1]:
                print(f"      ⚠︎ ПОСЛЕДНЕЕ слово: ждали «{want[-1]}», слышно «{got[-1]}»")
                rc = 1
        if pct < 80:
            print("      распознано:", " ".join(got[:40]))
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
