#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогноз с кэшем (3.8, п. 4): «Проба пера 3» второй раз - прогноз 2:19, факт 0:55.

Книга озвучена - в «_служебное/куски.json» список кусков. Часть кусков есть в
кэше синтеза - прогноз должен это знать; другой голос - «ничего не готово»;
файл книги поменялся - сведений нет. Без модели и без сети.

    python "Инструменты разработки/test_forecast_cache.py"
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import core.pipeline as P
from core import synthcache
from core.engines import get_engine

ok = True


def must(cond, msg):
    global ok
    print(("  ✓ " if cond else "  ✗ ") + msg)
    ok &= bool(cond)


tmp = tempfile.mkdtemp(prefix="herald-fc-")
synthcache.CACHE_DIR = os.path.join(tmp, "synth")
book = os.path.join(tmp, "Книга.txt")
open(book, "w", encoding="utf-8").write("Глава первая.\n\n" + "\n\n".join(
    f"Абзац номер {i}. Море к вечеру стихло, и над водой повис туман." for i in range(15)))
out = P.output_dir_for(book)
doc = P.read_document(book)
chapters = P.build_chapters(doc, 15)
prepared = [(c.title, P._chunk_paragraphs(c.text, 400, 0.4, 0.8, 1.1)) for c in chapters]
P.save_chunk_list(out, prepared, 0.4, book)
items = [(t, m) for _ci, (_ti, ch) in enumerate(prepared, 1)
         for t, _g, m, _k in P._norm_items(ch, 0.4)]
n = len(items)
ekey = get_engine("silero").cache_key()
vdir = P.default_voices_dir()
for t, m in items[: n * 3 // 4]:                       # три четверти кусков - в кэше
    synthcache.put(synthcache.key(ekey, vdir, "aidar", P._with_end_punct(t), m),
                   np.zeros(2400, np.float32), 24000, None)
got = P.cached_share(book, engine="silero", eng_kwargs={}, voice="aidar")
must(got is not None and got["n"] == n and got["cached"] == n * 3 // 4,
     f"видно, что {n * 3 // 4} из {n} кусков уже посчитаны: {got}")
other = P.cached_share(book, engine="silero", eng_kwargs={}, voice="eugene")
must(other is not None and other["cached"] == 0, "другой голос - ничего не готово")
with open(book, "a", encoding="utf-8") as f:
    f.write("\n\nНовый абзац.")
must(P.cached_share(book, engine="silero", eng_kwargs={}, voice="aidar") is None,
     "файл книги поменялся - старый список не годится")
# чтец в тегах (3.8, идея abogen): album_artist - имя голоса
import json as _json
import subprocess
from core.assemble import encode_chapter_mp3, RenderedChapter, write_wav
wav = os.path.join(tmp, "ч.wav")
dur = write_wav(np.zeros(24000, np.float32), 24000, wav)
mp3 = encode_chapter_mp3(RenderedChapter("Часть 1", wav, dur), os.path.join(tmp, "ч.mp3"),
                         title="Книга", author="Автор", narrator="Вячеслав Герасимов")
tags = _json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags",
                                   "-of", "json", mp3], capture_output=True, text=True).stdout)
tags = {k.lower(): v for k, v in tags.get("format", {}).get("tags", {}).items()}
must(tags.get("album_artist") == "Вячеслав Герасимов", f"чтец в тегах файла: {tags}")
import shutil
shutil.rmtree(tmp, ignore_errors=True)
print("OK: прогноз с кэшем, чтец в тегах" if ok else "ПРОВАЛ")
sys.exit(0 if ok else 1)
