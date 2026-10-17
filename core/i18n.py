"""Язык интерфейса RU | EN (3.7, стандарт интерфейса программ Андрея, п. 6).

Словари - в assets/i18n: ru - исходник (ключ - русская строка, как её видит
человек), en - перевод (en.json: {русская строка: английская}). В коде окна
надпись пишется через _("…"); при RU - как есть, при EN - из en.json. Ключ без
перевода ловит автотест (test_gui: «нет пропущенных ключей», «в окне нет
кириллицы при EN»).

Язык интерфейса не меняет язык книги или речи - их программа определяет сама.
По умолчанию - язык системы, выбор запоминается.
"""

from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DICT_DIR = os.path.join(HERE, "assets", "i18n")
LANGS = ("ru", "en")
_lang = "ru"
_en: dict[str, str] | None = None


def system_lang() -> str:
    """Язык системы: ru, если язык интерфейса системы русский, иначе en.
    Mac - первый язык из «Язык и регион» (AppleLanguages), Windows - язык
    интерфейса Windows, Linux - LANGUAGE / LC_ALL / LC_MESSAGES / LANG
    (C и POSIX - не язык, пропускаем)."""
    import sys
    try:
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["defaults", "read", "-g", "AppleLanguages"],
                                 capture_output=True, text=True, timeout=3).stdout
            first = re.search(r"[A-Za-z]{2}", out.replace("(", " "))
            if first:
                return "ru" if first.group(0).lower() == "ru" else "en"
        elif sys.platform == "win32":
            import ctypes
            return "ru" if ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF == 0x19 else "en"
    except Exception:
        pass
    for k in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        v = (os.environ.get(k) or "").split(":")[0]
        if v and not v.upper().startswith(("C", "POSIX")):
            return "ru" if v.lower().startswith("ru") else "en"
    return "en"


def set_lang(lang: str) -> None:
    global _lang
    _lang = lang if lang in LANGS else "ru"


def lang() -> str:
    return _lang


def _load_en() -> dict[str, str]:
    global _en
    if _en is None:
        try:
            _en = json.load(open(os.path.join(DICT_DIR, "en.json"), encoding="utf-8"))
        except (OSError, ValueError):
            _en = {}
    return _en


def _(text: str) -> str:
    """Надпись на языке интерфейса. Ключ - русская строка."""
    if _lang == "ru" or not text:
        return text
    return _load_en().get(text, text)


# Сообщения ядра о ходе работы - с числами внутри («часть 4 из 27»): переводятся
# по шаблонам при показе (ядро пишет журнал по-русски).
_PATTERNS: list[tuple[re.Pattern, str]] = []


def _patterns() -> list[tuple[re.Pattern, str]]:
    if not _PATTERNS:
        for ru, en in _load_en().get("__patterns__", {}).items() if isinstance(
                _load_en().get("__patterns__"), dict) else []:
            _PATTERNS.append((re.compile(ru), en))
    return _PATTERNS


def tr_msg(msg: str) -> str:
    """Сообщение ядра на языке интерфейса: целиком из словаря или по шаблону."""
    if _lang == "ru" or not msg:
        return msg
    d = _load_en()
    if msg in d:
        return d[msg]
    for rx, en in _patterns():
        if rx.search(msg):
            return rx.sub(en, msg)
    return msg


def text(name: str) -> str:
    """Длинный текст (справка, «О программе»): assets/i18n/<name>_<язык>.txt.
    «О программе» по-русски - из core.ABOUT (один исходник)."""
    path = os.path.join(DICT_DIR, f"{name}_{_lang}.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read()
    if name == "about":
        from . import ABOUT
        return ABOUT
    ru = os.path.join(DICT_DIR, f"{name}_ru.txt")
    return open(ru, encoding="utf-8").read() if os.path.exists(ru) else ""


def missing(keys) -> list[str]:
    """Ключи без английского перевода (для автотеста). Ключ без кириллицы
    («Vosk 4», «Joe») переводить не нужно."""
    d = _load_en()
    return [k for k in keys if k not in d and re.search("[А-Яа-яЁё]", k)]
