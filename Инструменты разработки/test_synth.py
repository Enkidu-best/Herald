#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Главная проверка: доходит ли текст до звука. Русский и английский.

Это тот тест, который надо гонять после ЛЮБОЙ правки. Он маленький (пара
предложений на язык) и отвечает на главный вопрос: записался файл или нет, и
то ли в нём, что просили. Всё остальное — украшения.

    python "Инструменты разработки/test_synth.py"
    python "Инструменты разработки/test_synth.py" --ru-voice "Игорь Ященко"
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import difflib
import os
import re
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault("PYTHONHASHSEED", "0")

MIN_WORD_MATCH = 75.0        # ниже этого считаем, что синтез сломан
MIN_SEC_PER_100_CHARS = 3.0  # меньше — значит файл пустой или обрезанный


def words(s: str, lang: str) -> list[str]:
    s = s.lower().replace("+", "").replace("ё", "е")
    keep = "а-яa-z" if lang == "ru" else "a-z"
    return re.sub(fr"[^{keep} ]", " ", s).split()


def check_one(book: str, voice: str, lang: str, out_dir: str) -> list[str]:
    from core.pipeline import convert_book, ConvertOptions
    from core.readers import read_document
    from core.chapters import build_chapters
    from core.normalize import normalize_text

    fails: list[str] = []
    name = os.path.basename(book)
    print(f"\n--- {name}  голосом «{voice}» ---")

    t0 = time.time()
    res = convert_book(book, engine="f5mlx", eng_kwargs={},
                       voice=voice, out_dir=out_dir,
                       options=ConvertOptions(minutes_per_file=15.0),
                       progress=lambda f, m: None, on_chapter=lambda *a: None)
    took = time.time() - t0

    if not res.mp3_dir or not os.path.isdir(res.mp3_dir):
        return [f"{name}: папка с результатом не создана"]
    files = sorted(f for f in os.listdir(res.mp3_dir) if not f.startswith("."))
    if not files:
        return [f"{name}: НИ ОДНОГО ФАЙЛА не записано"]

    path = os.path.join(res.mp3_dir, files[0])
    size = os.path.getsize(path)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout or 0)
    chapters = build_chapters(read_document(book))
    want = words(normalize_text(chapters[0].text, language=lang), lang)
    expect_sec = len(" ".join(want)) / 100 * MIN_SEC_PER_100_CHARS

    # прогноз не должен расходиться с фактом больше чем на четверть: по нему
    # человек решает, ставить книгу на ночь или нет
    from core.pipeline import estimate_book
    est = estimate_book(book, 15.0)
    err = abs(est.audio_sec - dur) / max(dur, 1) * 100
    print(f"  файл {files[0]}  {size//1024} КБ  {dur:.1f} c  (счёт {took:.0f} c, "
          f"на секунду звука {took/max(dur,1):.2f} с)")
    print(f"  прогноз звука {est.audio_sec:.0f} c против факта {dur:.0f} c — "
          f"расхождение {err:.0f}%")
    if err > 25:
        fails.append(f"{name}: прогноз звука врёт на {err:.0f}% "
                     f"({est.audio_sec:.0f} c против {dur:.0f} c)")
    if size == 0:
        fails.append(f"{name}: файл ПУСТОЙ (0 байт)")
    if dur < expect_sec:
        fails.append(f"{name}: звука всего {dur:.1f} c, ждали хотя бы {expect_sec:.0f} c")

    # Чистота звука. Проверяем ровно то, что однажды сломали и не заметили:
    # прибавка громкости с лимитером загоняла сигнал в потолок, лимитер начинал
    # работать непрерывно, и в речи появлялся треск. На слух это ловил только
    # слушатель, а числами видно сразу.
    import soundfile as _sf
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", path,
                    "-ac", "1", "-ar", "48000", "/tmp/_herald_check.wav"], check=True)
    snd, _sr = _sf.read("/tmp/_herald_check.wav")
    import numpy as _np
    snd = _np.asarray(snd, dtype=_np.float32)
    peak = float(_np.max(_np.abs(snd)))
    at_ceiling = float((_np.abs(snd) > 0.9).mean()) * 100
    print(f"  пик {peak:.2f} | сэмплов у потолка {at_ceiling:.3f}%")
    # Разделяет надёжно и не зависит от голоса: у файлов, которые слушатель
    # признал чистыми, пик 0,53-0,59 и НОЛЬ сэмплов у потолка; у тех, где он
    # услышал треск, пик 0,99 и 0,002-0,006% сэмплов прижаты к максимуму.
    if peak > 0.92:
        fails.append(f"{name}: сигнал у потолка (пик {peak:.2f}) — будет трещать")
    if at_ceiling > 0.001:
        fails.append(f"{name}: {at_ceiling:.3f}% сэмплов прижаты к максимуму — треск")

    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cpu", compute_type="int8")
    segs, _ = asr.transcribe(path, language=lang)
    got = words("".join(s.text for s in segs), lang)
    if not got:
        fails.append(f"{name}: в файле НЕ РАСПОЗНАНО НИ СЛОВА")
        return fails
    sm = difflib.SequenceMatcher(None, want, got)
    pct = 100.0 * sum(b.size for b in sm.get_matching_blocks()) / max(1, len(want))
    print(f"  слов совпало: {pct:.1f}%   «{' '.join(got[:9])}…»")
    if pct < MIN_WORD_MATCH:
        fails.append(f"{name}: совпало только {pct:.0f}% слов")
    # Края проверяем с допуском в два слова: распознаватель и сам может
    # разделить или склеить слово на границе, а нам важно, что оно ЗВУЧИТ,
    # а не стоит ли ровно первым в расшифровке.
    def near(word: str, among: list[str]) -> bool:
        # одна смазанная буква на стыке — не потеря слова, а предел модели
        return any(difflib.SequenceMatcher(None, word, g).ratio() >= 0.7
                   for g in among)
    if not near(want[0], got[:3]):
        fails.append(f"{name}: первого слова «{want[0]}» не слышно "
                     f"(начало: {' '.join(got[:3])})")
    if not near(want[-1], got[-3:]):
        fails.append(f"{name}: последнего слова «{want[-1]}» не слышно "
                     f"(конец: {' '.join(got[-3:])})")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru-voice", default=None)
    ap.add_argument("--en-voice", default=None)
    ap.add_argument("--keep", action="store_true", help="не удалять готовые файлы")
    ap.add_argument("--quick", action="store_true",
                    help="только короткие примеры (без проверки громких мест)")
    args = ap.parse_args()

    from core.engines import get_engine
    voices = get_engine("f5mlx").list_voices()
    ru = args.ru_voice or next((v.id for v in voices if v.language == "ru"), None)
    en = args.en_voice or next((v.id for v in voices if v.language == "en"), None)

    out = tempfile.mkdtemp(prefix="herald_test_")
    fails: list[str] = []
    # Короткие примеры — про то, доходит ли текст до звука. Длинный обязателен
    # отдельно: громкие места в книге редки, на 13 секундах их можно не
    # встретить, и клиппинг проходит мимо теста (так и случилось).
    books = [("Примеры текстов/Короткий тест — ru.txt", ru, "ru"),
             ("Примеры текстов/Короткий тест — en.txt", en, "en")]
    if not args.quick:
        books.append(("Примеры текстов/Пример — проба пера.txt", ru, "ru"))
    for book, voice, lang in books:
        if not voice:
            print(f"\n--- {os.path.basename(book)}: нет голоса для «{lang}», пропуск")
            continue
        try:
            fails += check_one(book, voice, lang, out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            fails.append(f"{os.path.basename(book)}: {type(e).__name__}: {e}")

    print(f"\nрезультат в {out}" if args.keep else "")
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)

    print()
    if fails:
        print(f"ПРОВАЛЕНО {len(fails)}:")
        for f in fails:
            print("   ", f)
        return 1
    print("оба языка озвучиваются, текст на месте")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
