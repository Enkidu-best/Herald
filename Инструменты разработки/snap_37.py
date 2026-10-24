#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимки окна 3.7 для docs/screenshots: RU и EN, светлая и тёмная тема,
все разделы, масштаб 120 %.

Окно поднимается поверх полноэкранного VS Code (fullScreenAuxiliary) и
снимается по номеру окна: screencapture -l<CGWindowID>. Настройки окна
после съёмки возвращаются как были.

    caffeinate -dimsu .venv/bin/python "Инструменты разработки/snap_37.py" docs/screenshots

В docs/screenshots снимки лежат в JPEG шириной 1416 (sips -Z 1416, качество 82).
"""
import copy, os, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "docs/screenshots")
os.makedirs(out, exist_ok=True)

import customtkinter as ctk
import theme as T
import tts_gui as G
from core import i18n
import Quartz, AppKit

T.apply()
cfg = copy.deepcopy(G.load_config())
# снимки - при масштабе по умолчанию (120 %): и виджеты, и размер окна; настройки
# Андрея возвращаются в конце
G.save_config({**cfg, G.SCALE_KEY: G.DEFAULT_SCALE})
ctk.set_widget_scaling(G.DEFAULT_SCALE)
BOOKS = {"ru": "Примеры текстов/Пример - проба пера.txt",
         "en": "Примеры текстов/Example - Tide and Lantern.txt"}
SAY = {"ru": "Вода у берега была тёмной, и фонарь на мачте едва светил. Он взялся за вёсла.",
       "en": "The water near the shore was dark, and the lantern on the mast barely glowed."}


def snap_all(lang):
    i18n.set_lang(lang)
    T.reset_icons()
    app = G.App()
    app.update()

    def pump(sec):
        end = time.time() + sec
        while time.time() < end:
            app.update()
            time.sleep(0.05)

    def wid():
        pid = os.getpid()
        for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
            if w.get("kCGWindowOwnerPID") == pid and w.get("kCGWindowName") == G.APP_NAME \
                    and w.get("kCGWindowBounds", {}).get("Height", 0) > 300:
                return w["kCGWindowNumber"]

    def shot(name):
        # к AppKit - только когда фоновые потоки окна отработали (ошибка GIL в pyobjc)
        end = time.time() + 30
        while threading.active_count() > 1 and time.time() < end:
            pump(0.1)
        pump(0.3)
        for w in AppKit.NSApp.windows():
            w.setCollectionBehavior_(1 | 256)
            w.setLevel_(3)
            w.orderFrontRegardless()
        pump(0.8)
        p = os.path.join(out, name + ".png")
        r = subprocess.run(["screencapture", "-x", "-o", f"-l{wid()}", p], capture_output=True, text=True)
        print(name, "->", p if os.path.exists(p) else r.stderr.strip(), flush=True)

    app._set_book(BOOKS[lang])
    t0 = time.time()
    while app.est is None and time.time() - t0 < 30:
        pump(0.2)
    app.show("say")
    app.say_box.delete("1.0", "end")
    app.say_box.insert("1.0", SAY[lang])
    for mode, word in (("light", "light"), ("dark", "dark")):
        app.set_theme(mode)
        pump(0.5)
        for page in ("book", "say", "voices", "settings", "help"):
            app.show(page)
            shot(f"{lang} - {page} - {word}")
    app.destroy()


try:
    snap_all("ru")
    snap_all("en")
finally:
    G.save_config(cfg)
    i18n.set_lang("ru")
os._exit(0)
