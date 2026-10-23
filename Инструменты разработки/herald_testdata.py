# -*- coding: utf-8 -*-
"""Своя папка данных для проверок Herald (стандарт 6.10, задание 13 Б).

Импортировать ДО `core` и `tts_gui`: переменная HERALD_DATA_DIR читается при импорте.
Проверка получает пустую временную папку данных (настройки, голоса, корзина, поправки,
кэш синтеза, журнал) и сама кладёт туда то, что ей нужно:
  * два клона: «Тест Клон» (образец + расшифровка) и «Тест Обучен» (ещё и свои веса
    `.f5model` - пустой файл весов во временной папке); голос Qwen «Тест Диктор»
    (`voices/qwen/`, виден, только где Qwen установлен);
  * пример книги «Пример - проба пера.txt» (свой текст, не из «Примеры текстов/»).
Модели (Silero, F5, meter2) берутся из обычной папки моделей - они не данные Андрея.
Данные Андрея (`~/.cache/text2audio`: голоса, настройки, корзина, поправки) - снимок
хэшей до проверки и сравнение после: `andrey_changed()`.

    import herald_testdata as TD      # первая строка проверки
    TD.setup()                        # -> TD.ROOT, TD.BOOK, TD.CLONE, TD.TRAINED
"""
from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = ""
BOOK = ""
CLONE = "Тест Клон"
TRAINED = "Тест Обучен"
QWEN = "Тест Диктор"
ANDREY = Path.home() / ".cache" / "text2audio"
MAC = sys.platform == "darwin"
_before: dict[str, str] = {}

BOOK_TEXT = """Проба пера

Глава первая

Море к вечеру стихло, и над водой повис туман. Старый смотритель маяка поднялся
по винтовой лестнице - сто двадцать семь ступеней - и зажёг огонь.

В 1912 году здесь разбился пароход «Надежда»; спаслись 14 человек из 40. С тех пор
смотритель каждый вечер проверял лампу дважды.

- Опять туман, - сказал он коту. - Значит, ночью никто не придёт.

Кот промолчал. Где-то далеко прогудел буксир, и снова стало тихо.

Глава вторая

Утром на берег вынесло доску с надписью на чужом языке. Смотритель долго вертел её
в руках, потом отнёс в сарай, к остальным находкам: бутылке, веслу и старому замку.
"""

SAMPLE_TEXT = "Море к вечеру стихло, и над водой повис туман. Старый смотритель поднялся наверх."


def _tone(path: str, seconds: float = 8.0, f0: float = 120.0) -> None:
    """Образец без модели: гармоники с огибающей слогов - похоже на голос для отбора."""
    import numpy as np
    import soundfile as sf
    sr = 24000
    t = np.arange(int(sr * seconds)) / sr
    a = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t).clip(0, 1)
    sf.write(path, (0.2 * a * env / 3).astype("float32"), sr)


def _snapshot() -> dict[str, str]:
    """Хэши данных Андрея, которые проверка могла бы задеть (без кэша синтеза и моделей)."""
    out = {}
    for rel in ("herald.json", "book_fixes.json", "трудные_ударения.json"):
        p = ANDREY / rel
        if p.is_file():
            out[rel] = hashlib.sha1(p.read_bytes()).hexdigest()
    for sub in ("voices", "trash"):
        d = ANDREY / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(ANDREY))] = hashlib.sha1(p.read_bytes()).hexdigest()
    return out


def setup(prefix: str = "herald_test_", *, clones: bool = True, book: bool = True) -> str:
    """Временная папка данных; HERALD_DATA_DIR - на неё. Повторный вызов - та же папка."""
    global ROOT, BOOK, _before
    if ROOT:
        return ROOT
    _before = _snapshot()
    ROOT = tempfile.mkdtemp(prefix=prefix)
    os.environ["HERALD_DATA_DIR"] = os.path.join(ROOT, "data")
    os.environ.setdefault("HERALD_MODEL_DIR", str(ANDREY))    # модели - общие, только чтение
    os.environ["HERALD_HEAVY_LOCK"] = os.path.join(ROOT, ".heavy-lock")   # не настоящая очередь
    os.environ.pop("HERALD_HEAVY_LOCK_HELD", None)
    vd = os.path.join(ROOT, "data", "voices")
    os.makedirs(vd)
    os.makedirs(os.path.join(ROOT, "data", "trash"))
    if clones:
        for name, f0 in ((CLONE, 120.0), (TRAINED, 105.0)):
            _tone(os.path.join(vd, name + ".wav"), f0=f0)
            open(os.path.join(vd, name + ".txt"), "w", encoding="utf-8").write(SAMPLE_TEXT)
        w = os.path.join(ROOT, "weights")
        os.makedirs(w)
        open(os.path.join(w, "model_v1.safetensors"), "wb").close()
        open(os.path.join(vd, TRAINED + ".f5model"), "w", encoding="utf-8").write(w)
        qd = os.path.join(vd, "qwen")
        os.makedirs(qd)
        _tone(os.path.join(qd, QWEN + ".wav"), f0=180.0)
        open(os.path.join(qd, QWEN + ".txt"), "w", encoding="utf-8").write(SAMPLE_TEXT)
    if book:
        os.makedirs(os.path.join(ROOT, "Книги"))
        BOOK = os.path.join(ROOT, "Книги", "Пример - проба пера.txt")
        open(BOOK, "w", encoding="utf-8").write(BOOK_TEXT)
    atexit.register(lambda: shutil.rmtree(ROOT, ignore_errors=True))
    return ROOT


def andrey_changed() -> list[str]:
    """Файлы данных Андрея, которые изменились с начала проверки (пусто - ничего)."""
    now = _snapshot()
    return sorted(k for k in set(_before) | set(now) if _before.get(k) != now.get(k))


def need_mac(why: str) -> str | None:
    """Причина пропуска на не-Mac («нужен Mac: …») или None, если можно проверять."""
    return None if MAC else f"нужен Mac: {why}"
