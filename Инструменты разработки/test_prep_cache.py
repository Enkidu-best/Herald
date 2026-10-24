#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""14 Г: подготовленный текст книги (числа, ударения) берётся из «.herald» при «Продолжить» и повторном запуске.

  1. кэш байт в байт равен подготовке заново - текст частей, спорные слова, решения ударений (2 книги из «Примеры
     текстов», копии во временной папке - рядом с оригиналами лежат прошлые озвучки);
  2. второй запуск не размечает ни одной главы; изменилась одна глава - заново только она;
  3. изменилась подготовка (код, словари, модели) или её настройки - заново всё;
  4. «.herald» уходит - кэша нет (готовая книга, «Начать заново»).
  5. --book <файл>: замер шага 1 до / после на копии книги (только чтение, без синтеза).

    .venv/bin/python "Инструменты разработки/test_prep_cache.py" [--book <файл книги>]
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import herald_testdata as TD  # noqa: E402

TD.setup(book=False)
from core import pipeline as P, stress  # noqa: E402
from core.chapters import build_chapters  # noqa: E402
from core.normalize import detect_language  # noqa: E402
from core.readers import read_document  # noqa: E402

SAMPLES = HERE.parent / "Примеры текстов"
BOOKS = ("Пример - проба пера 2.txt", "Пример - проба пера 3.txt")
FAILS: list[str] = []


def check(name: str, ok: bool, note: str = "") -> None:
    print(("  OK    " if ok else "  FAIL  ") + name + (f" - {note}" if note and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def chapters_of(path: str, opts):
    doc = read_document(path)
    chs = build_chapters(doc, opts.minutes_per_file)
    return chs, detect_language(doc.text or "")


def fresh(chs, lang, opts):
    """Подготовка, как до 14 Г: все главы подряд, спорное и решения - после."""
    stress.take_disputed()
    stress.take_decisions()
    bodies = [P._prepare_text(ch.text, "+", opts.expand_numbers, opts.use_stress, lang) for ch in chs]
    return bodies, stress.take_disputed(), stress.take_decisions()


def cached(chs, lang, opts, workdir):
    bodies, hits = P._prepare_chapters(chs, workdir, "+", opts.expand_numbers, opts.use_stress, lang)
    return bodies, stress.take_disputed(), stress.take_decisions(), hits


class Count:
    """Сколько глав размечено заново (обёртка _prepare_text)."""

    def __init__(self):
        self.n, self.orig = 0, P._prepare_text

    def __enter__(self):
        def wrap(*a, **k):
            self.n += 1
            return self.orig(*a, **k)
        P._prepare_text = wrap
        return self

    def __exit__(self, *exc):
        P._prepare_text = self.orig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="herald-prep-14g-"))
    opts = P.ConvertOptions(minutes_per_file=2)          # несколько частей и у короткой книги
    try:
        for name in BOOKS:
            src = tmp / name
            shutil.copy(SAMPLES / name, src)
            chs, lang = chapters_of(str(src), opts)
            work = os.path.join(P.output_dir_for(str(src)), P.STATE_DIR)
            print(f"{name}: частей {len(chs)}")
            b0, d0, r0 = fresh(chs, lang, opts)
            with Count() as c1:
                b1, d1, r1, h1 = cached(chs, lang, opts, work)
            check("первый запуск - всё размечено, кэш записан", c1.n == len(chs) and h1 == 0
                  and os.path.exists(os.path.join(work, P.PREP_FILE)))
            with Count() as c2:
                b2, d2, r2, h2 = cached(chs, lang, opts, work)
            check("второй запуск - ни одной главы заново", c2.n == 0 and h2 == len(chs), f"заново {c2.n}")
            check("текст частей байт в байт как подготовка заново", b0 == b1 == b2)
            check("спорные ударения те же", d0 == d1 == d2, f"{len(d0)} / {len(d1)} / {len(d2)}")
            check("решения ударений те же (и число мест)", r0 == r1 == r2)
            if len(chs) > 1:
                chs[-1].text = chs[-1].text + "\nДописанная строка: в 1990 году он вернулся в замок."
                with Count() as c3:
                    b3, *_ = cached(chs, lang, opts, work)
                check("изменилась одна глава - заново только она", c3.n == 1, f"заново {c3.n}")
                check("остальные главы - те же", b3[:-1] == b0[:-1])
            env = P._prep_env
            P._prep_env = lambda: "другая подготовка"
            try:
                with Count() as c4:
                    cached(chs, lang, opts, work)
            finally:
                P._prep_env = env
            check("изменилась подготовка (код, словари, модели) - заново всё", c4.n == len(chs))
            o2 = P.ConvertOptions(minutes_per_file=2, expand_numbers=not opts.expand_numbers)
            with Count() as c5:
                cached(chs, lang, o2, work)
            check("другие настройки подготовки - заново всё", c5.n == len(chs))
            P.forget_progress(str(src))
            with Count() as c6:
                cached(chs, lang, opts, work)
            check("«Начать заново» (нет .herald) - заново всё", c6.n == len(chs))
            # «Как будет прочитано» - тот же кэш
            t1 = P.prepare_read_text(str(src), opts)
            with Count() as c7:
                t2 = P.prepare_read_text(str(src), opts)
            check("«Как будет прочитано» - из кэша, текст тот же", c7.n == 0 and t1 == t2, f"заново {c7.n}")
        if a.book:
            src = tmp / Path(a.book).name
            shutil.copy(a.book, src)                     # книга Андрея - только копия
            o = P.ConvertOptions()
            chs, lang = chapters_of(str(src), o)
            work = os.path.join(P.output_dir_for(str(src)), P.STATE_DIR)
            t0 = time.time()
            b1, *_ = cached(chs, lang, o, work)
            t_first = time.time() - t0
            t0 = time.time()
            b2, *_ = cached(chs, lang, o, work)
            t_second = time.time() - t0
            print(f"замер: «{src.name}», частей {len(chs)}, знаков {sum(len(c.text) for c in chs)}: шаг 1 - "
                  f"{t_first:.1f} с, «Продолжить» с кэшем - {t_second:.2f} с")
            check("книга: кэш = подготовка", b1 == b2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    check("данные Андрея не изменились", not TD.andrey_changed())
    print("всё в порядке" if not FAILS else f"ошибок: {len(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
