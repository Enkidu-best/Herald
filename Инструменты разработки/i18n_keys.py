#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ключи словаря интерфейса (3.7): все надписи окна и сообщения ядра.

Явные _("…") в коде окна + значения, которые окно переводит при показе
(названия движков и режимов, подписи пауз, длины частей, форматы…) + сообщения
ядра. Печатает ключи без перевода в assets/i18n/en.json (--missing) или пишет
их списком (--out).

    python "Инструменты разработки/i18n_keys.py" --missing
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import ast
import json


def code_keys(path: str) -> list[str]:
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_"
                and n.args and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.append(n.args[0].value)
    return out


def all_keys() -> list[str]:
    import tts_gui as G
    from core import profiles, osutil, APP_TAGLINE, stream
    from core.assemble import FORMATS
    keys = code_keys("tts_gui.py")
    keys += [p.title for p in profiles.PROFILES.values()] + [p.hint for p in profiles.PROFILES.values()]
    keys += [profiles.SILERO.title] + list(profiles.ACCEL_TITLE.values())
    keys += [q.title for q in profiles.QUALITY.values()] + [q.hint for q in profiles.QUALITY.values()]
    keys += [t for _k, t, *_r in G.PAUSES] + list(G.GENDER.values()) + list(G.LANG_MARK.values())
    from core import voice_store as VS
    keys += [v.title for v in G.SILERO_VOICES] + [v.title for v in VS.CATALOG]  # имена готовых голосов
    keys += list(G.FILE_LENGTHS) + list(FORMATS) + [APP_TAGLINE, "на этом Mac", "на этом компьютере"]
    # сообщения ядра, которые показывает окно (ход работы, состояние потока, ошибки)
    keys += ["Читаю файл…", "Готовлю текст (числа, ударения)…", "Склеиваю части в один файл…",
             "Собираю аудиокнигу .m4b…", "Готово", "Озвучиваю пробный фрагмент…", "Пробник готов",
             "Скачиваю модель проверки слов (один раз, 1,3 ГБ)…", "Готовлю модель синтеза…", "Проверяю, как модель ставит ударения (один раз для модели, ~3 мин)…",
             "Скачиваю модели (первый запуск, около 4,5 ГБ - это может занять полчаса)…",
             "Готовлю текст…", "Играет · считаю дальше", "Пауза · считаю дальше",
             "Играет · всё посчитано", "Пауза · всё посчитано", "Остановлено",
             "Звук потоком не вышел - подробности в журнале",
             "Нет связи с интернетом - голос не скачался. Проверьте подключение и нажмите "
             "«Скачать» ещё раз.", "загрузка отменена"]
    seen, out = set(), []
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keys = all_keys()
    if a.out:
        json.dump(keys, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.missing:
        from core import i18n
        miss = i18n.missing(keys)
        print(f"ключей {len(keys)}, без перевода {len(miss)}")
        for k in miss[:50]:
            print("  ", repr(k[:90]))
        return 1 if miss else 0
    print(len(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
