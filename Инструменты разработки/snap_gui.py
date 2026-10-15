#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Снимки окна в обеих темах для docs/Приёмка 3.1 (Озвучка, Голоса, Настройки).

Окно поднимается поверх полноэкранного VS Code (fullScreenAuxiliary) и
снимается по номеру окна: screencapture -l<CGWindowID>.

    caffeinate -dimsu .venv/bin/python "Инструменты разработки/snap_gui.py" docs/снимки
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

# книга и «идущая работа» для «Озвучки»
book = "Книги/Жажда власти 1 - Сергей Тармашев/Жажда власти. Книга 1.fb2"
if not os.path.exists(book):
    book = "Примеры текстов/Пример - проба пера.txt"
app._set_book(book)
t0 = time.time()
while app.est is None and time.time() - t0 < 30:
    pump(0.2)
d = os.path.dirname(book)
for f in sorted(x for x in os.listdir(d) if x.startswith("Часть "))[:4]:
    app._add_part(os.path.splitext(f)[0], os.path.join(d, f), "")
names = {"light": "светлая", "dark": "тёмная"}
# «Голоса» и «Настройки» - в обычном состоянии
for mode in ("light", "dark"):
    app.set_theme(mode); pump(0.5)
    app.show("voices"); pump(0.3)
    me = cfg.get("voice") or next(v.id for v in app._all_voices() if not v.ready and v.language == "ru")
    app.pick_voice(me); pump(0.3)
    # наведение на соседнюю строку - чтобы на снимке было видно подсветку
    other = next(k for k in app._voice_row_widgets if k != me)
    app._voice_row_widgets[other].configure(fg_color=app._voice_row_color(other, True))
    shot(f"голоса - {names[mode]}")
    app.show("settings"); shot(f"настройки - {names[mode]}")
# «Озвучка» - в момент работы: этап, части, кнопки
app._set_busy(True, "")
app.pause_btn.configure(state="normal")
app.progress.set(0.22)
app.part_progress.set(0.4)
app.part_label.configure(text="Часть 5 из 27 - 40%")
app._t0 = time.time() - (3 * 3600 + 12 * 60)
app.status.configure(text="Шаг 2 из 3 · Озвучка - часть 5 из 27, 22 % · осталось ~11 ч 20 мин")
app._progress = lambda *a: None
app._live_rtf = 2.05
app.status.configure(text="Шаг 2 из 3 · Озвучка - часть 5 из 27, 22 % · осталось ~11 ч 20 мин · RTF 2,05")
rows = [w for w in app.rows.winfo_children() if isinstance(w, G.PartRow)]
if rows:                                   # мини-плеер у первой части, на паузе
    app.listen(rows[0]); pump(0.3)
    app.player.pause(); app.player.seek(72.0)
for mode in ("light", "dark"):
    app.set_theme(mode); pump(0.5)
    app.show("book")
    if rows and rows[0].transport is None:
        rows[0].open_transport(app.player)
    rows and rows[0].sync(app.player)
    shot(f"озвучка - {names[mode]}")
app._close_player()
app.running = False
G.save_config(cfg)
ctk.set_appearance_mode(cfg.get('theme', 'system'))
app.destroy()
