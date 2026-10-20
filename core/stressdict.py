"""Справочник ударений по формам слов - данные Викисловаря (11а, п. 3).

Где RUAccent и silero-stress разошлись на обычном слове («щавель», «красивее»),
спор решает справочник: 862 тыс. словоформ Викисловаря с ударением и грамматикой
формы («стены» - сте́ны им. / вин. п. мн. ч., стены́ род. п. ед. ч.). Это
встроенная часть программы: пользователь его не ведёт и не видит.

Данные - `core/data/stress_wiktionary.tsv.gz` (Викисловарь, CC BY-SA 3.0, через
tsnorm 1.1.2) и `stress_zaliznyak.tsv.gz` (по «Грамматическому словарю» Зализняка,
CC0); см. core/data/README.md. При первом обращении текст перекладывается
в SQLite в папке данных (около 10 с, один раз): поиск слова - доли миллисекунды,
в памяти ничего не держим.
"""

from __future__ import annotations

import gzip
import os
import sqlite3
import threading

from .log import log_problem

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stress_wiktionary.tsv.gz")
VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
VERSION = "1"                      # поменять, если поменяется файл данных

_lock = threading.Lock()
_db: sqlite3.Connection | None = None
_tried = False


def _path() -> str:
    from .osutil import data_dir
    d = os.path.join(str(data_dir()), "stressdict")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"wiktionary-{VERSION}.sqlite")


def key(w: str) -> str:
    return w.replace("+", "").replace("́", "").lower().replace("ё", "е")


def build(dst: str | None = None, src: str = SRC) -> str:
    dst = dst or _path()
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    db = sqlite3.connect(tmp)
    db.execute("CREATE TABLE f (k TEXT, v TEXT, tags TEXT)")
    with gzip.open(src, "rt", encoding="utf-8") as f:
        rows = (line.rstrip("\n").split("\t") for line in f if not line.startswith("#"))
        db.executemany("INSERT INTO f VALUES (?, ?, ?)", (r for r in rows if len(r) == 3))
    db.execute("CREATE INDEX fk ON f (k)")
    db.commit()
    db.close()
    os.replace(tmp, dst)
    return dst


def _get() -> sqlite3.Connection | None:
    global _db, _tried
    if _db is not None or _tried:
        return _db
    with _lock:
        if _db is not None or _tried:
            return _db
        _tried = True
        try:
            p = _path()
            if not os.path.exists(p):
                build(p)
            _db = sqlite3.connect(p, check_same_thread=False)
        except Exception as e:
            log_problem("справочник ударений недоступен - спорные слова решает RUAccent", e, show=False)
            _db = None
    return _db


def variants(word: str) -> dict[str, set[str]]:
    """Все ударения формы: {«ст+ены»: {"им", "мн", …}, «стен+ы»: {"род", "ед", …}}."""
    db = _get()
    if db is None:
        return {}
    out: dict[str, set[str]] = {}
    with _lock:
        for v, tags in db.execute("SELECT v, tags FROM f WHERE k = ?", (key(word),)):
            out.setdefault(v, set()).update(tags.split())
    return out


def stress_of(marked: str) -> tuple:
    """Номера ударных гласных в форме с «+» (ё - ударная)."""
    idx, vi, plus = [], -1, False
    for c in marked:
        if c == "+":
            plus = True
            continue
        if c in VOWELS:
            vi += 1
            if plus or c in "ёЁ":
                idx.append(vi)
        plus = False
    return tuple(sorted(set(idx)))


def options(word: str) -> dict[tuple, set[str]]:
    """Разные ударения формы (номера гласных) -> теги грамматики."""
    out: dict[tuple, set[str]] = {}
    for v, tags in variants(word).items():
        s = stress_of(v)
        if s:
            out.setdefault(s, set()).update(tags)
    return out


ZAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "stress_zaliznyak.tsv.gz")
_zal: dict[str, tuple] | None = None


def zaliznyak(word: str) -> tuple | None:
    """Ударение формы по «Грамматическому словарю» Зализняка (108 тыс. форм, CC0),
    если оно там одно, иначе None. Словарь - нормативный: при споре он первый."""
    global _zal
    if _zal is None:
        with _lock:
            if _zal is None:
                z: dict[str, tuple] = {}
                try:
                    with gzip.open(ZAL, "rt", encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("#"):
                                continue
                            k, v = line.rstrip("\n").split("\t")
                            if "," not in v:
                                z[k] = (int(v),)
                except Exception as e:
                    log_problem("словарь Зализняка не прочитался", e, show=False)
                _zal = z
    return _zal.get(key(word))


def unique(word: str) -> tuple | None:
    """Ударение формы, если справочник знает ровно одно: сначала Зализняк, потом
    Викисловарь (там бывают варианты: «щ+авель» и «щав+ель»); иначе None."""
    z = zaliznyak(word)
    if z:
        return z
    o = options(word)
    return next(iter(o)) if len(o) == 1 else None


def available() -> bool:
    return _get() is not None
