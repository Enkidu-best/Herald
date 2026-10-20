#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка ударений: то, что уже однажды ломалось молча.

  * фраза с именем (слово вне словаря) получает ударения — раньше из-за
    несовместимости RUAccent с transformers 5.x такие фразы уходили в синтез
    совсем без ударений (на «Жажде власти» — 19 % фраз);
  * омографы различаются по контексту;
  * «ё» доходит до синтеза;
  * слово вне словаря звучит одинаково во всей книге.

    python "Инструменты разработки/test_stress.py"
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fails = []


def check(name, ok, got=""):
    print(f"  {'OK  ' if ok else 'ПЛОХО'}  {name}" + (f"   ({got})" if not ok else ""))
    if not ok:
        fails.append(name)


def main() -> int:
    from core.normalize import normalize_text
    from core.stress import add_stress_text, unify_book, add_stress
    s = add_stress_text("Фринштадт стоял на холме, и мама мыла раму.")
    check("фраза с именем вне словаря получает ударения", "м+ама" in s and "+" in s.split()[0], s)
    a = add_stress_text("Ключ легко открыл замок на двери.")
    b = add_stress_text("Над рекой возвышался старинный замок.")
    check("омограф: замОк", "зам+ок" in a, a)
    check("омограф: зАмок", "з+амок" in b, b)
    y = add_stress_text(normalize_text("Её ещё нет, но всё будет готово.", language="ru"))
    check("«ё» доходит до синтеза", all(w in y for w in ("Её", "ещё", "всё")), y)
    t = [add_stress("Ланд-капитан вошёл. Ланд-капитан сел."),
         add_stress("Тенкатль кивнул ланд-капитану, ланд-капитан ответил.")]
    out, rep = unify_book(t)
    import re
    forms = {w for x in out for w in re.findall(r"[Лл]\+?анд-капит\+?ан\b", x)}
    check("слово вне словаря одинаково по книге", len({f.lower() for f in forms}) == 1, forms)
    # 11а: длинный абзац - по предложениям (раньше склейка до 2 500 знаков и сбой на 512 токенах)
    para = ("Связи требует канцелярия Владыки Императорского Доминиона, желают получить представление "
            "о положении дел. Подождут. " * 6 +
            "Допустим, что это было сделано в качестве жеста доброй воли по отношению к Малому "
            "Доминиону Чиалори, к которому принадлежат владельцы добывающей компании. "
            "Больше всего сенаторов у Малого Доминиона Чиалори.")
    p = add_stress_text(para)
    check("длинный абзац: «Св+язи требует»", p.startswith("Св+язи"), p[:40])
    check("длинный абзац: «к М+алому», «у М+алого»", "к М+алому" in p and "у М+алого" in p,
          re.findall(r"\S+ М\+?ал\+?ог?о?му?", p))
    check("длинный абзац: у каждого предложения ударения",
          all("+" in s for s in re.split(r"(?<=[.!?])\s+", p) if len(s) > 12), "")
    # 11а: спор разметчиков на обычном слове - словарь (Зализняк, Викисловарь)
    w = add_stress_text("Красивее этого места на побережье не было.")
    check("красивее: крас+ивее (по предложению, без подмены silero)", "Крас+ивее" in w, w)
    w = add_stress_text("Расскажи им что-нибудь интересное.")
    check("словарь: чт+о-нибудь", "чт+о-нибудь" in w, w)
    from core.stress import _token_variant
    check("вариант «всё» в регистре текста", _token_variant("Все", "вс+ё") == "Вс+ё",
          _token_variant("Все", "вс+ё"))
    print("всё в порядке" if not fails else f"ПРОВАЛЕНО: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
