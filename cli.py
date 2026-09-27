#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Текст в Аудио — консольный запуск (без GUI).

Примеры:
    # разово подготовить голос-образец (обрезка + расшифровка) и озвучить пробник
    python3 cli.py "Книга.fb2" --engine f5 --ref "образец.mp3" --preview 1

    # вся книга голосом ранее подготовленного образца
    python3 cli.py "Книга.fb2" --engine f5 --voice образец

    # быстрый запасной движок
    python3 cli.py "Книга.epub" --engine silero --voice eugene --mp3

    python3 cli.py --list-voices --engine silero
"""

from __future__ import annotations

import os
# Некоторые зависимости (ctranslate2/faster-whisper, multiprocessing) падают,
# если PYTHONHASHSEED задан пустым/битым. Фиксируем валидное значение до импортов.
_hs = os.environ.get("PYTHONHASHSEED")
if _hs is None or not _hs.isdigit():
    os.environ["PYTHONHASHSEED"] = "0"
# MPS: разрешаем откат неподдержанных операций на CPU (иначе падение на видеоядре)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import sys

from core.engines import get_engine, list_engines
from core.engines.f5 import F5Engine
from core.pipeline import convert_book, preview_sample, ConvertOptions
from core.readers import ScannedPdfError, ReaderError


def _progress(fr: float, msg: str) -> None:
    bar = int(fr * 30)
    sys.stderr.write("\r[" + "#" * bar + "." * (30 - bar) + f"] {int(fr*100):3d}%  {msg}   ")
    sys.stderr.flush()
    if fr >= 1.0:
        sys.stderr.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Озвучка текста в аудиокнигу (локально)")
    ap.add_argument("inputs", nargs="*", help="файлы: .txt .md .docx .epub .fb2 .pdf")
    ap.add_argument("--engine", default="f5mlx",
                    help="движок синтеза: f5mlx (быстрый, по умолчанию) | f5 (torch, медленный) | silero")
    ap.add_argument("--voice", default=None, help="голос (Silero) или id образца (F5)")
    ap.add_argument("--ref", default=None, help="F5: аудио-образец голоса (подготовит и использует)")
    ap.add_argument("--preview", type=float, default=0, metavar="МИН",
                    help="сделать пробник на N минут из середины главы (не всю книгу)")
    ap.add_argument("--chapter", type=int, default=None, help="номер главы для пробника (с 1)")
    ap.add_argument("--out", default=None, help="каталог для результата")
    ap.add_argument("--m4b", action="store_true", help="ещё собрать единый файл-аудиокнигу .m4b (по умолч. только mp3 по главам)")
    ap.add_argument("--no-mp3", action="store_true", help="не сохранять отдельные mp3 по главам")
    ap.add_argument("--no-numbers", action="store_true", help="не разворачивать числа в слова")
    ap.add_argument("--no-stress", action="store_true", help="не расставлять ударения")
    ap.add_argument("--speed", type=float, default=None, help="F5: темп речи (>1 быстрее; ~1.2 по умолч.)")
    ap.add_argument("--nfe", type=int, default=None,
                    help="шагов синтеза (меньше=быстрее). f5mlx: 7 по умолчанию, ещё 6/10/12/16")
    ap.add_argument("--schedule", default=None,
                    help="f5mlx: epss (прорежённое расписание, по умолчанию) | uniform (как torch)")
    ap.add_argument("--workers", type=int, default=1, help="параллельных процессов для глав (ускорение на многоядерном M4)")
    ap.add_argument("--device", default=None,
                    help="только f5 (torch): cpu. mps на 16 ГБ упирается в память и медленнее — не советую")
    ap.add_argument("--list-voices", action="store_true", help="показать голоса движка")
    args = ap.parse_args()

    if args.list_voices:
        eng = get_engine(args.engine)
        print(f"Движок {args.engine}:")
        for v in eng.list_voices():
            print(f"    {v.id:20s} {v.title}")
        return 0

    # движок
    eng_kw = {}
    if args.engine == "f5":
        if args.speed is not None:
            eng_kw["speed"] = args.speed
        if args.nfe is not None:
            eng_kw["nfe_step"] = args.nfe
        if args.device is not None:
            eng_kw["device"] = args.device
    elif args.engine == "f5mlx":
        if args.speed is not None:
            eng_kw["speed"] = args.speed
        if args.nfe is not None:
            eng_kw["steps"] = args.nfe
        if args.schedule is not None:
            eng_kw["schedule"] = args.schedule
    voice = args.voice
    if args.ref:
        eng_ref = get_engine(args.engine, **eng_kw)
        if isinstance(eng_ref, F5Engine):
            print(f"Готовлю образец голоса: {os.path.basename(args.ref)} …", file=sys.stderr)
            voice = eng_ref.prepare_reference(args.ref)
            print(f"  голос сохранён как: {voice}", file=sys.stderr)
    if not voice:
        print("Не задан голос: укажите --voice, а для F5 — --ref <образец> или --voice <id>.",
              file=sys.stderr)
        return 1
    if not args.inputs:
        ap.print_help()
        return 1

    opts = ConvertOptions(make_m4b=args.m4b, make_chapter_mp3=not args.no_mp3,
                          expand_numbers=not args.no_numbers, use_stress=not args.no_stress,
                          workers=args.workers)
    rc = 0
    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"Пропуск (нет файла): {path}", file=sys.stderr); rc = 1; continue
        print(f"\n=== {os.path.basename(path)} ===")
        try:
            import time, subprocess
            import soundfile as sf
            import numpy as np
            t0 = time.time()
            if args.preview > 0:
                ci = (args.chapter - 1) if args.chapter else None
                out = preview_sample(path, engine=args.engine, eng_kwargs=eng_kw, voice=voice,
                                     minutes=args.preview, chapter_index=ci, options=opts,
                                     progress=_progress)
                elapsed = time.time() - t0
                a, sr = sf.read(out)
                dur = len(a) / sr if len(a) else 0.0
                rms = float(np.sqrt(np.mean(np.square(a)))) if len(a) else 0.0
                base = os.path.splitext(os.path.basename(path))[0]
                mp3 = os.path.join(args.out or os.path.dirname(os.path.abspath(path)),
                                   base + " — пробник.mp3")
                # CBR + Xing-заголовок -> корректная длительность в любом плеере
                from core.assemble import polish_wav_to_mp3
                polish_wav_to_mp3(out, mp3)
                print(f"  Пробник: {mp3}")
                print(f"  Длительность: {dur:.1f} c | громкость(RMS): {rms:.4f} | "
                      f"счёт на этом компьютере: {elapsed:.0f} c")
                if rms < 1e-3:
                    print("  ВНИМАНИЕ: похоже, тишина (RMS≈0). Пришли этот вывод — разберёмся.")
            else:
                def _on_ch(ci, title, mp3):
                    sys.stderr.write("\n")
                    if mp3:
                        print(f"  ✓ глава {ci} готова: {os.path.basename(mp3)}  (можно слушать)")
                    else:
                        print(f"  ✓ глава {ci} готова")
                res = convert_book(path, engine=args.engine, eng_kwargs=eng_kw, voice=voice,
                                   out_dir=args.out, options=opts, progress=_progress,
                                   on_chapter=_on_ch)
                elapsed = time.time() - t0
                mins = int(res.duration // 60)
                print(f"  Глав: {res.chapters}, длительность ~{mins} мин | "
                      f"счёт: {elapsed:.0f} c")
                if res.m4b_path:
                    print(f"  Аудиокнига: {res.m4b_path}")
                if res.mp3_dir:
                    print(f"  Главы (mp3): {res.mp3_dir}")
        except ScannedPdfError as e:
            print(f"  ПРОПУЩЕН: {e}", file=sys.stderr); rc = 1
        except ReaderError as e:
            print(f"  ОШИБКА чтения: {e}", file=sys.stderr); rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
