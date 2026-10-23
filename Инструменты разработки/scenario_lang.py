#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сценарий 3.7: RU | EN, «≡» и «Aa» в настоящем окне (через main(), как у
пользователя). Окно пересобирается - книга и раздел остаются, выбор
запоминается. Данные - свои (временная папка, `herald_testdata.py`): настройки
Андрея не трогаются.

    python "Инструменты разработки/scenario_lang.py"
"""
import copy, os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
os.chdir(sys.path[0])
import herald_testdata as TD          # до tts_gui: HERALD_DATA_DIR
TD.setup("herald_lang_")

import tts_gui as G
from core import i18n

BOOK = TD.BOOK
backup = copy.deepcopy(G.load_config())
G.save_config({**backup, "ui_lang": "ru"})
seen, problems = [], []
real_init = G.App.__init__


def check(cond, msg):
    if not cond:
        problems.append(msg)


def init(self):
    real_init(self)
    k = len(seen) + 1

    def step():
        try:
            seen.append((i18n.lang(), self.page, self.crumb.cget("text"),
                         self.cfg.get("menu_collapsed", False)))
            if k == 1:
                self._set_book(BOOK)
                self.show("settings")
                self._on_lang("EN")
            elif k == 2:
                check(i18n.lang() == "en" and self.lang_seg.get() == "EN", "язык не сменился")
                check(self.page == "settings" and self.crumb.cget("text") == "Settings",
                      f"раздел потерялся: {self.page}, {self.crumb.cget('text')!r}")
                check(G.load_config().get("ui_lang") == "en", "выбор языка не запомнился")
                self.show("book")
                check(self.book_path == BOOK, "книга потерялась после смены языка")
                check(self.crumb.cget("text").startswith("Narration › "),
                      f"путь в верхней строке: {self.crumb.cget('text')!r}")
                self._toggle_menu()
            elif k == 3:
                check(self.cfg.get("menu_collapsed") is True, "меню не свернулось")
                check(G.VERSION in self.version_lbl.cget("text"), "версии внизу нет")
                self._set_scale(1.3)
                check(abs(G.load_config().get("scale", 0) - 1.3) < 1e-6, "масштаб не запомнился")
                self._set_scale(G.DEFAULT_SCALE)
                self._toggle_menu()
            elif k == 4:
                self._on_lang("RU")
            else:
                check(i18n.lang() == "ru" and self.crumb.cget("text").startswith("Озвучка"),
                      "обратно на русский не вернулось")
                self.quit()
        except Exception:
            problems.append(traceback.format_exc())
            self.quit()
    self.after(1200, step)


G.App.__init__ = init
_exit = os._exit


def finish(code):
    G.save_config(backup)
    for s in seen:
        print("  окно:", s)
    ok = not problems and len(seen) == 5
    print("всё в порядке" if ok else "ПРОВАЛЕНО:\n  " + "\n  ".join(problems or ["окон " + str(len(seen))]))
    sys.stdout.flush()
    _exit(0 if ok else 1)


os._exit = finish
try:
    G.main()
finally:
    finish(1)
