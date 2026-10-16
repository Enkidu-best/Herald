#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимки вкладки «Прочитать вслух» в обеих темах (3.6): поле, подсветка, голос.

Окно поднимается поверх полноэкранного VS Code (fullScreenAuxiliary) и
снимается по номеру окна: screencapture -l<CGWindowID>.

    caffeinate -dimsu .venv/bin/python "Инструменты разработки/snap_say.py" docs/снимки
"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(sys.path[0])
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "docs/снимки")
os.makedirs(out, exist_ok=True)

import customtkinter as ctk
import theme as T
import tts_gui as G
import Quartz, AppKit

T.apply()
cfg = G.load_config()
app = G.App()
app.geometry("1180x780+60+40")
app.update()

def front():
    for w in AppKit.NSApp.windows():
        w.setCollectionBehavior_(1 | 256)
        w.setLevel_(3)
        w.orderFrontRegardless()

def wid():
    pid = os.getpid()
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") == pid and w.get("kCGWindowName") == G.APP_NAME \
                and w.get("kCGWindowBounds", {}).get("Height", 0) > 300:
            return w["kCGWindowNumber"]

def pump(sec):
    end = time.time() + sec
    while time.time() < end:
        app.update()
        time.sleep(0.05)

def shot(name):
    # обращаться к AppKit можно только когда фоновые потоки окна отработали:
    # иначе pyobjc падает с ошибкой GIL (так и случилось при первой пересъёмке)
    import threading
    end = time.time() + 30
    while threading.active_count() > 1 and time.time() < end:
        pump(0.1)
    pump(0.3)
    front(); pump(0.8)
    n = wid()
    p = os.path.join(out, name + ".png")
    r = subprocess.run(["screencapture", "-x", "-o", f"-l{n}", p], capture_output=True, text=True)
    print(name, "->", p if os.path.exists(p) else r.stderr.strip())

text = open("Примеры текстов/Пример - проба пера.txt", encoding="utf-8").read().split("\n\n")
text = "\n\n".join(p for p in text[1:4] if p.strip())
app.show("say"); pump(0.3)
app.say_box.delete("1.0", "end")
app.say_box.insert("1.0", text)
app._say_text_changed()
app._say_on_kind("Быстрый", remember=False)
eug = next(v for v in app._voices if v.id == "eugene")
app.say_voice_var.set(app._say_label(eug))
app._say_on_voice(remember=False)
spans = G.S.sentence_spans(app._say_text())
app._say_highlight(spans[1])                 # «звучит» второе предложение
app.say_note.configure(text="Играет · считаю дальше")
names = {"light": "светлая", "dark": "тёмная"}
for mode in ("light", "dark"):
    app.set_theme(mode); pump(0.5)
    app.show("say"); app._say_highlight(spans[1]); pump(0.3)
    shot(f"прочитать вслух - {names[mode]}")
G.save_config(cfg)
ctk.set_appearance_mode(cfg.get('theme', 'system'))
app.destroy()
