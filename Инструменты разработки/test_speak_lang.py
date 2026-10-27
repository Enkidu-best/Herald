#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2а (ночь 23-24.10): /speak не отдаёт голосу текст чужого алфавита (common/tts_client). Без модели.

    python "Инструменты разработки/test_speak_lang.py"
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.speak_server import voice_text  # noqa: E402

fails = []


def check(name, ok, note=""):
    print(("  OK    " if ok else "  ПАДАЕТ ") + name + (f": {note}" if note and not ok else ""), flush=True)
    if not ok:
        fails.append(name)


t, sk = voice_text({"text": "Answer: используйте STAR", "fast": True})
check("быстрый (русский) голос: только русское", t == "используйте" and not re.search(r"[A-Za-z]", t), t)
check("английские отрывки - в пропущенных", sk == [("en", "Answer:"), ("en", "STAR")], str(sk))
t, sk = voice_text({"text": "Сократили срок на 30 % за 2 дня.", "fast": True})
check("русская фраза - целиком", t == "Сократили срок на 30 % за 2 дня." and not sk, t)
t, sk = voice_text({"text": "Say it in English. Скажите по-русски.", "fast": False, "lang": "en"})
check("клон английского голоса - только английское", t == "Say it in English." and sk == [("ru", "Скажите по-русски.")], t)
t, sk = voice_text({"text": "STAR", "fast": True})
check("нет слов языка голоса - пусто (сервис ответит ошибкой, не по буквам)", t == "" and sk == [("en", "STAR")], t)
from core.speak_server import voices_list  # noqa: E402
v = voices_list()
check("GET /voices: 5 быстрых голосов Silero с подписями", [x["id"] for x in v["fast"]] == ["aidar", "eugene", "baya", "kseniya", "xenia"]
      and all(x["title"] for x in v["fast"]), str(v["fast"]))
check("GET /voices: клоны - списком (без загрузки моделей)", isinstance(v["clones"], list), str(v["clones"])[:80])
print(f"\nИтог: {7 - len(fails)} из 7")
sys.exit(1 if fails else 0)
