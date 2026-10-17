#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Herald - окно программы (3.0), в стиле Scribe 2.3 / Legal Knowledge Desk.

Боковое меню: Озвучка · Голоса · Настройки · Тема.
«Озвучка»: слева книга и готовые части («Слушать», «Показать в Finder»),
справа параметры, внизу шкала с этапом словами, оставшимся временем и большой
кнопкой «Озвучить книгу» / «Остановить». Технический журнал свёрнут.

Правило потоков: фоновые потоки НЕ трогают окно. Всё, что нужно показать,
кладётся в очередь self.q и выполняется главным потоком (self._drain). Настройки
снимаются с виджетов ДО запуска потока. Раньше потоки читали переменные Tk
напрямую - это работало, пока главный цикл крутился, и падало, когда нет.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
import tkinterdnd2

import theme as T
from core import VERSION, APP_NAME, APP_TAGLINE, ABOUT
from core import keepawake, profiles, osutil
from core import stream as S
from core.assemble import FORMATS, DEFAULT_FORMAT, polish_wav_to_mp3, HD_FILTER
from core.chapters import part_name
from core.engines import get_engine, clone_engine, accel_label
from core.engines.f5 import default_voices_dir
from core.engines.silero import VOICES as SILERO_VOICES
from core.log import log_problem, log_info, LOG_PATH
from core.pipeline import (convert_book, preview_sample, estimate_book, ConvertOptions,
                           Cancelled, resume_info, forget_progress, cleanup_stale_temp,
                           output_dir_for)
from core.readers import ScannedPdfError, ReaderError
from core.voicefx import VoiceFX, load as load_fx, save as save_fx
from player import Player, fmt_time

BOOK_EXT = (".txt", ".md", ".docx", ".epub", ".fb2", ".zip", ".pdf")
SINGLE_FILE = "Один файл (без частей)"
REVEAL_TEXT = osutil.REVEAL_TEXT
STRICT_COST = 1.55         # строгая проверка слов: счёт дольше (замер 3.2: +50-58 %)
SOUND_COST = 1.05          # проверка по звуку: с 3.5 края тоже по звуку - почти даром
FILE_LENGTHS = ["10 минут", "15 минут", "20 минут", "30 минут", "45 минут",
                "60 минут", SINGLE_FILE]
# При «Одном файле» синтез всё равно идёт частями по столько минут - ради
# потоковой выдачи и чтобы «Стоп» не терял сделанное; склейка в конце.
SINGLE_PART_MINUTES = 15.0
CONFIG = os.path.expanduser("~/.cache/text2audio/herald.json")
LANG_MARK = {"ru": "русский", "en": "английский"}
# Паузы (секунды): ключ настройки, подпись, по умолчанию, от, до
PAUSES = (("gap_sec", "После предложения", 0.4, 0.2, 1.5),
          ("para_gap_sec", "После абзаца", 0.8, 0.3, 2.0),
          ("dialog_gap_sec", "Между репликами", 1.1, 0.3, 2.5))
VOICE_NAME_LIMIT = 24
FAST_VOICE = "aidar"       # «Прочитать вслух» быстрым голосом: готовый голос Silero
SAY_HL_BG, SAY_HL_FG = "#FFE99A", "#1A1A1A"   # подсветка звучащего предложения
SAY_TEMPO = (0.5, 1.5)     # темп во вкладке (3.6: Андрей хотел медленнее 0,8)
GENDER = {"м": "муж.", "ж": "жен."}


# --- мелочи ----------------------------------------------------------------
def _restress(word: str, plus: str) -> str:
    """Слово (с комбинируемым ударением) - с ударением из «plus», регистр - как был."""
    letters = [c for c in word if c != "\u0301"]
    out, j = [], 0
    for ch in plus:
        if ch == "+":
            continue
        c = letters[j] if j < len(letters) else ch
        out.append(c)
        j += 1
        if j - 1 < len(plus) and _stressed_at(plus, j - 1):
            out.append("\u0301")
    return "".join(out)


def _stressed_at(plus: str, letter_idx: int) -> bool:
    """Стоит ли ударение на букве номер letter_idx в слове с «+»."""
    k = -1
    for i, ch in enumerate(plus):
        if ch == "+":
            continue
        k += 1
        if k == letter_idx:
            return i > 0 and plus[i - 1] == "+"
    return False


def _clone_engine(engine: str) -> bool:
    """Клон F5 (темп в модель, долгий старт) или готовый быстрый голос."""
    return engine in ("f5mlx", "f5torch", "f5")


def _open_path(path: str):
    osutil.open_path(path)


def _reveal(path: str):
    osutil.reveal(path)


def _fmt_hms(sec: float) -> str:
    """«1 ч 05 мин», «3 мин 56 с», «42 с»."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return f"{h} ч {m:02d} мин"
    if m:
        return f"{m} мин {s:02d} с"
    return f"{s} с"


def _fmt_clock(sec: float) -> str:
    """Живой таймер: «3 мин 07 с», а за час - «1 ч 05 мин 12 с»."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h} ч {m:02d} мин {s:02d} с" if h else f"{m} мин {s:02d} с"


def _files(n: int) -> str:
    """«1 файл», «3 файла», «12 файлов»."""
    if n % 10 == 1 and n % 100 != 11:
        w = "файл"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        w = "файла"
    else:
        w = "файлов"
    return f"{n} {w}"


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1] + "…"


def load_config() -> dict:
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log_problem("настройки окна не прочитались, беру обычные", e, show=False)
        return {}


def save_config(cfg: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        with open(CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
    except OSError as e:
        log_problem("настройки окна не сохранились", e, show=False)


class _V:
    """Голос в списке: клон (из образца) или готовый (Silero)."""

    def __init__(self, vid: str, title: str, language: str, ready: bool,
                 engine: str = "silero", gender: str = ""):
        self.id, self.title, self.language, self.ready = vid, title, language, ready
        self.engine, self.gender = engine, gender

    def label(self) -> str:
        name = _short(self.title, VOICE_NAME_LIMIT)
        return f"{name} · {LANG_MARK.get(self.language, self.language)}"


# --- строка готовой части ----------------------------------------------------
class PartRow(ctk.CTkFrame):
    """Готовая часть: «Слушать» открывает в строке мини-плеер.

    Плеер один на окно (App.player); строка лишь показывает его кнопки:
    играть/пауза одной кнопкой, «стоп», ползунок перемотки и время.
    """

    def __init__(self, master, title: str, path: str | None, note: str = "",
                 on_listen=None):
        super().__init__(master, fg_color="transparent", corner_radius=0)
        self.path = path
        self.on_listen = on_listen
        self.transport = None
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text="", image=T.icon("audio-lines", T.MUTED, 18)).grid(
            row=0, column=0, rowspan=2, padx=(14, 10), pady=(10, 0), sticky="n")
        T.label(self, _short(title, 60), 14, bold=True, anchor="w").grid(
            row=0, column=1, sticky="ew", pady=(8, 0))
        T.label(self, note, 12, color=T.MUTED, anchor="w").grid(
            row=1, column=1, sticky="ew", pady=(0, 8))
        acts = ctk.CTkFrame(self, fg_color="transparent")
        acts.grid(row=0, column=2, rowspan=2, padx=(8, 10))
        if path:
            self.listen_btn = T.button(
                acts, "Слушать", "play", height=28, size=13, width=96,
                command=lambda: self.on_listen(self) if self.on_listen else _open_path(path))
            self.listen_btn.pack(side="left", padx=(0, 4))
            # одна кнопка папки на часть (3.2): «Показать в Finder» рядом с плеером
            self.reveal_btn = T.button(acts, REVEAL_TEXT, "folder-open", kind="ghost",
                                       height=28, size=13, command=lambda: _reveal(path))
            self.reveal_btn.pack(side="left")
        else:
            T.chip(acts, "не записалась", "bad").pack()
        ctk.CTkFrame(self, height=1, fg_color=T.BORDER, corner_radius=0).grid(
            row=3, column=0, columnspan=3, sticky="ew")

    def open_transport(self, player: Player):
        if self.transport is not None:
            return
        tr = ctk.CTkFrame(self, fg_color=T.SURFACE2, corner_radius=8)
        tr.grid(row=2, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 10))
        tr.grid_columnconfigure(2, weight=1)
        self.play_btn = T.button(tr, "", "pause", kind="primary", height=30, width=36,
                                 command=lambda: (player.toggle(), self.sync(player)))
        self.play_btn.grid(row=0, column=0, padx=(8, 4), pady=6)
        self.stop_btn = T.button(tr, "", "square", height=30, width=36,
                                 command=lambda: (player.stop(), self.sync(player)))
        self.stop_btn.grid(row=0, column=1, padx=(0, 8), pady=6)
        self.seek = ctk.CTkSlider(tr, from_=0, to=1, number_of_steps=1000, height=16,
                                  button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
                                  progress_color=T.ACCENT, fg_color=T.BORDER,
                                  command=lambda v: (player.seek(v * player.duration),
                                                     self.sync(player)))
        self.seek.grid(row=0, column=2, sticky="ew", pady=6)
        self.time_lbl = T.label(tr, "0:00 / 0:00", 12, color=T.MUTED, width=110, anchor="e")
        self.time_lbl.grid(row=0, column=3, padx=(8, 10))
        self.transport = tr
        self.sync(player)

    def close_transport(self):
        if self.transport is not None:
            self.transport.destroy()
            self.transport = None

    def sync(self, player: Player):
        """Привести кнопки и ползунок к состоянию плеера."""
        if self.transport is None:
            return
        playing = player.playing
        if getattr(self, "_shown_playing", None) != playing:
            T.restyle(self.play_btn, "primary", "pause" if playing else "play")
            self._shown_playing = playing
        dur = player.duration
        pos = player.position
        self.seek.set(pos / dur if dur else 0)
        self.time_lbl.configure(text=f"{fmt_time(pos)} / {fmt_time(dur)}")


class App(ctk.CTk, tkinterdnd2.TkinterDnD.DnDWrapper):
    FX_PHRASE = {
        "ru": "Дом сто+ял на краю дер+евни, и к+аждое +утро над ним подним+ался д+ым.",
        "en": "The house stood at the edge of the village, and smoke rose above it every morning.",
    }

    def __init__(self):
        super().__init__()
        self.TkdndVersion = tkinterdnd2.TkinterDnD._require(self)
        self.cfg = load_config()
        self.title(APP_NAME)
        self.geometry("1180x780")
        self.minsize(1040, 700)
        try:
            import tkinter as _tk
            ic = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
            if os.path.exists(ic):
                self.iconphoto(True, _tk.PhotoImage(file=ic))
        except Exception:
            pass                       # иконка окна - только украшение

        self.q: queue.Queue = queue.Queue()
        self.running = False
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._paused_at = 0.0
        self._t0 = 0.0
        self._synth_t0 = None
        self._synth_fr = 0.0
        self._step = 0
        self.book_path: str | None = None
        self.est = None
        self._resume = None
        self._voices: list[_V] = []
        self.page = None
        self._fx_raw = None
        self._fx_voice = None
        self._fx_vars: dict = {}
        self.player = Player()
        self._active_row: PartRow | None = None
        self._live_rtf: float | None = None
        self._pause_vars: dict = {}
        self._accel = "mlx" if osutil.IS_APPLE_SILICON else "cpu"
        # звук потоком (core/stream): пробник, «Прослушать», «Прочитать вслух».
        # Одновременно звучит один; новый останавливает прежний
        self._live = None
        self._live_kind = ""
        self._say_last = None             # (звук, частота, поправки) для «Сохранить»
        self._fx_player = None
        self._prof_btns: dict = {}
        self._srv = None                  # озвучка для других программ (core/speak_server)
        self._srv_voice = None            # голос и движок для сервиса - копия из окна:
        self._srv_kwargs: dict = {}       # переменные tkinter из чужого потока не читаем

        self._build_menu()
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        ctk.CTkFrame(self, width=1, fg_color=T.BORDER, corner_radius=0).grid(
            row=0, column=1, sticky="ns")
        self.content = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0)
        self.content.grid(row=0, column=2, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)
        # страницы строим заранее: построение в момент щелчка по меню давало мигание
        self.pages = {"book": self._build_book_page(),
                      "say": self._build_say_page(),
                      "voices": self._build_voices_page(),
                      "settings": self._build_settings_page()}
        self.show("book")
        self.refresh_voices()
        self._say_on_kind(self.say_mode.get(), remember=False)
        self._srv_sync()
        if self.cfg.get("speak_server"):
            self.after(300, self._srv_apply)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.drop_target_register(tkinterdnd2.DND_FILES)
        self.dnd_bind("<<Drop>>", self.on_drop)
        self.after(100, self._drain)
        threading.Thread(target=cleanup_stale_temp, daemon=True).start()

        def detect():                     # torch грузится секунды - не в главном потоке
            try:
                a = accel_label()
            except Exception as e:
                log_problem("не удалось определить, чем считать озвучку", e, show=False)
                a = "cpu"
            self.ui(lambda: self._show_accel(a))
        threading.Thread(target=detect, daemon=True).start()

    def on_close(self):
        """Закрытие окна: звук мини-плеера и поток останавливаются вместе с окном."""
        self._close_player()
        self._live_stop()
        if self._srv is not None:
            self._srv.stop()
        self.destroy()

    # ================= связь потоков с окном =================
    def ui(self, fn):
        """Выполнить fn в главном потоке (из любого потока)."""
        self.q.put(fn)

    def _drain(self):
        try:
            while True:
                fn = self.q.get_nowait()
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.after(100, self._drain)

    # ================= меню и боковая панель =================
    def _build_menu(self):
        """Меню «Правка»: без него на macOS Cmd+C в окне не работает вовсе."""
        import tkinter as tk
        menubar = tk.Menu(self)
        edit = tk.Menu(menubar, tearoff=0)
        edit.add_command(label="Скопировать", accelerator="Cmd+C",
                         command=lambda: self._menu_event("<<Copy>>"))
        edit.add_command(label="Выделить всё", accelerator="Cmd+A",
                         command=lambda: self._menu_event("<<SelectAll>>"))
        edit.add_separator()
        edit.add_command(label="Скопировать подробности", command=lambda: self.copy_log())
        menubar.add_cascade(label="Правка", menu=edit)
        self.configure(menu=menubar)

    def _menu_event(self, event: str):
        w = self.focus_get()
        if w is not None:
            w.event_generate(event)

    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=220, fg_color=T.SURFACE, corner_radius=0)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        side.pack_propagate(False)
        T.label(side, APP_NAME, 19, bold=True, color=T.ACCENT, anchor="w").pack(
            fill="x", padx=22, pady=(22, 2))
        T.label(side, APP_TAGLINE, 12, color=T.MUTED, anchor="w").pack(
            fill="x", padx=22, pady=(0, 18))
        self.nav = {}
        for key, text, ic in (("book", "Озвучка", "audio-lines"), ("say", "Прочитать вслух", "play"),
                              ("voices", "Голоса", "mic")):
            self.nav[key] = self._nav_btn(side, text, ic, key)
            self.nav[key].pack(fill="x", padx=12, pady=1)
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=12, pady=(0, 14))
        ctk.CTkFrame(bottom, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=(0, 8))
        self.nav["settings"] = self._nav_btn(bottom, "Настройки", "settings", "settings")
        self.nav["settings"].pack(fill="x", pady=1)
        self.theme_btn = self._nav_btn(bottom, "", "sun-moon", None)
        self.theme_btn.configure(command=self._cycle_theme)
        self.theme_btn.pack(fill="x", pady=1)
        self._theme_label()
        T.label(bottom, f"{APP_NAME} {VERSION}\nлокально {osutil.HERE}", 11, color=T.FAINT,
                anchor="w", justify="left").pack(fill="x", padx=10, pady=(8, 0))

    def _nav_btn(self, parent, text, ic, key):
        b = ctk.CTkButton(parent, text=text, anchor="w", height=36, corner_radius=8, border_width=0,
                          border_color=T.SURFACE, fg_color="transparent", hover_color=T.SURFACE2,
                          text_color=T.TEXT, text_color_disabled=T.FAINT, font=T.font(14),
                          image=T.icon(ic, T.MUTED, 18), compound="left",
                          command=(lambda: self.show(key)) if key else None)
        b._ic = ic
        return b

    def show(self, key):
        # сначала перекрасить меню и дать ему отрисоваться, потом показать
        # страницу - иначе пункт меню на миг мигает чужим цветом (опыт Scribe)
        for k, b in self.nav.items():
            on = k == key
            b.configure(fg_color=T.ACCENT_SOFT if on else "transparent",
                        hover_color=T.ACCENT_SOFT if on else T.SURFACE2,
                        border_color=T.ACCENT_SOFT if on else T.SURFACE,
                        text_color=T.ACCENT if on else T.TEXT, font=T.font(14, bold=on),
                        image=T.icon(b._ic, T.ACCENT if on else T.MUTED, 18))
        self.update_idletasks()
        for k, p in self.pages.items():
            if k == key:
                p.grid(row=0, column=0, sticky="nsew")
            else:
                p.grid_remove()
        self.page = key
        # Прокручиваемые списки customtkinter теряют отрисовку, если тему
        # сменили, пока страница была скрыта: строки на месте, но не показаны
        # (поймано на снимках тёмной темы). Событие <Configure> их возвращает.
        self.after(40, self._redraw_lists)
        if key == "voices":
            self._refresh_voice_rows()

    def _redraw_lists(self):
        for sf in (self.rows, self.voice_rows, self.settings_scroll):
            try:
                sf._parent_canvas.event_generate("<Configure>")
            except Exception:
                pass                # список ещё не построен - нечего чинить

    def _theme_label(self):
        names = {"system": "Системная", "light": "Светлая", "dark": "Тёмная"}
        self.theme_btn.configure(text=f"Тема: {names.get(self.cfg.get('theme', 'system'), 'Системная')}")

    def _cycle_theme(self):
        order = ["system", "light", "dark"]
        cur = self.cfg.get("theme", "system")
        nxt = order[(order.index(cur) + 1) % 3] if cur in order else "light"
        self.set_theme(nxt)

    def set_theme(self, mode: str):
        self.cfg["theme"] = mode
        ctk.set_appearance_mode(mode)
        self._theme_label()
        save_config(self.cfg)
        if self.page:
            self.show(self.page)            # перерисовать видимую страницу

    # ================= страница «Озвучка» =================
    def _build_book_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=28, pady=(22, 14))
        T.label(head, "Озвучка", 22, bold=True, anchor="w").pack(fill="x")
        T.label(head, "Книга - в аудиокнигу голосом выбранного диктора. Всё считается "
                      f"{osutil.HERE}; части появляются рядом с книгой по мере готовности.",
                13, color=T.MUTED, anchor="w").pack(fill="x", pady=(2, 0))

        left = ctk.CTkFrame(page, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(28, 8))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.book_card = T.card(left)
        self.book_card.grid(row=0, column=0, sticky="nsew")
        self._build_empty(self.book_card)
        self._build_book_info(self.book_card)
        self._build_details(left)

        opts = T.card(page)
        opts.grid(row=1, column=1, sticky="nsew", padx=(8, 28))
        self._build_options(opts)

        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=28, pady=(14, 20))
        lft = ctk.CTkFrame(bar, fg_color="transparent")
        lft.pack(side="left", fill="x", expand=True, padx=(0, 18))
        prow = ctk.CTkFrame(lft, fg_color="transparent")
        prow.pack(fill="x")
        self.progress = ctk.CTkProgressBar(prow, height=6)
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True, pady=(4, 0))
        self.timer = T.label(prow, _fmt_clock(0), 13, color=T.MUTED, width=130, anchor="e")
        self.timer.pack(side="right", padx=(10, 0))
        self.status = T.label(lft, "Перетащите книгу - и нажмите «Озвучить книгу».", 13,
                              color=T.MUTED, anchor="w")
        self.status.pack(fill="x", pady=(4, 0))
        prow2 = ctk.CTkFrame(lft, fg_color="transparent")
        prow2.pack(fill="x", pady=(4, 0))
        self.part_label = T.label(prow2, "", 12, color=T.FAINT, anchor="w", width=230)
        self.part_label.pack(side="left")
        self.part_progress = ctk.CTkProgressBar(prow2, height=4, progress_color=T.FAINT)
        self.part_progress.set(0)
        self.part_progress.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.restart_btn = T.button(bar, "Начать заново", "refresh-cw", kind="ghost", height=42,
                                    size=13, width=150, command=self.do_restart)
        self.restart_btn.configure(border_width=1, border_color=T.BORDER_STRONG)
        self.pause_btn = T.button(bar, "Пауза", "pause", height=42, size=14, width=130,
                                  command=self.do_pause)
        self.pause_btn.configure(state="disabled")
        self.pause_btn.pack(side="right", padx=(8, 0))
        self.start_btn = T.button(bar, "Озвучить книгу", "play", kind="primary", width=230,
                                  height=42, size=15, command=self.on_main_button)
        self.start_btn.pack(side="right")
        return page

    def _build_empty(self, parent):
        self.empty = ctk.CTkFrame(parent, fg_color=T.SURFACE, border_width=2,
                                  border_color=T.BORDER_STRONG, corner_radius=T.RADIUS)
        self.empty.pack(fill="both", expand=True, padx=14, pady=14)
        inner = ctk.CTkFrame(self.empty, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ic = ctk.CTkLabel(inner, text="", image=T.icon("upload", T.ACCENT, 36))
        ic.pack(pady=(0, 10))
        title = T.label(inner, "Перетащите книгу сюда", 17, bold=True)
        title.pack()
        sub = T.label(inner, "txt · md · docx · epub · fb2 · pdf", 13, color=T.MUTED,
                      justify="center")
        sub.pack(pady=(4, 14))
        T.button(inner, "Выбрать книгу", "plus", kind="primary", width=170,
                 command=self.choose_book).pack()
        for w in (self.empty, inner, ic, title, sub):
            w.bind("<Button-1>", lambda e: self.choose_book())

    def _build_book_info(self, parent):
        self.info = ctk.CTkFrame(parent, fg_color="transparent")
        top = ctk.CTkFrame(self.info, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(14, 6))
        top.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(top, text="", image=T.icon("file-text", T.ACCENT, 28)).grid(
            row=0, column=0, rowspan=3, padx=(0, 12), sticky="n")
        self.book_title = T.label(top, "", 17, bold=True, anchor="w", justify="left",
                                  wraplength=460)
        self.book_title.grid(row=0, column=1, sticky="ew")
        self.book_author = T.label(top, "", 13, color=T.MUTED, anchor="w")
        self.book_author.grid(row=1, column=1, sticky="ew")
        self.est_label = T.label(top, "", 13, anchor="w", justify="left", wraplength=460)
        self.est_label.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        self.clear_btn = T.button(top, "", "x", kind="ghost", height=28, command=self.clear_book)
        self.clear_btn.grid(row=0, column=2, sticky="ne")
        ctk.CTkFrame(self.info, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=(8, 0))
        hdr = ctk.CTkFrame(self.info, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(10, 6))
        self.parts_title = T.label(hdr, "Готовые части", 15, bold=True, anchor="w")
        self.parts_title.pack(side="left")
        self.rows = ctk.CTkScrollableFrame(self.info, fg_color="transparent", corner_radius=0)
        self.rows.pack(fill="both", expand=True, padx=1, pady=(0, 6))
        self.no_parts = T.label(self.rows, "Пока ни одной. Части появятся здесь по мере "
                                           "готовности - их можно слушать сразу.",
                                12, color=T.MUTED, anchor="w", justify="left", wraplength=460)

    def _build_details(self, parent):
        """Технический журнал - свёрнутый блок «Подробности»."""
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.details_box = box
        if not self.cfg.get("show_log", False):
            self.after(10, box.grid_remove)
        line = ctk.CTkFrame(box, fg_color="transparent")
        line.pack(fill="x")
        self.details_btn = T.button(line, "Подробности", "file-text", kind="ghost", height=28,
                                    size=13, width=140, command=self.toggle_details)
        self.details_btn.pack(side="left")
        self.copy_btn = T.button(line, "Копировать", "copy", kind="ghost", height=28, size=13,
                                 width=130, command=self.copy_log)
        self.log = ctk.CTkTextbox(box, height=150, font=T.font(12), border_width=1)
        self._enable_copy(self.log)
        self._details_open = False

    def toggle_details(self):
        self._details_open = not self._details_open
        if self._details_open:
            self.copy_btn.pack(side="left", padx=(6, 0))
            self.log.pack(fill="x", pady=(6, 0))
            self.log.see("end")
        else:
            self.copy_btn.pack_forget()
            self.log.pack_forget()

    def _build_options(self, card):
        o = ctk.CTkFrame(card, fg_color="transparent")
        o.pack(fill="both", expand=True, padx=18, pady=16)
        ctk.CTkFrame(o, width=290, height=0, fg_color="transparent").pack()
        T.section(o, "Голос").pack(fill="x")
        self.voice_var = ctk.StringVar(value="")
        self.voice_menu = ctk.CTkOptionMenu(o, variable=self.voice_var, values=[""], height=32,
                                            font=T.font(13), dropdown_font=T.font(13),
                                            command=lambda *_: self._on_voice())
        self.voice_menu.pack(fill="x", pady=(6, 2))
        self.voice_hint = T.label(o, "", 12, color=T.MUTED, anchor="w", justify="left",
                                  wraplength=290)
        self.voice_hint.pack(fill="x")

        T.section(o, "Длина частей").pack(fill="x", pady=(16, 0))
        self.len_var = ctk.StringVar(value=self.cfg.get("len", "15 минут")
                                     if self.cfg.get("len") in FILE_LENGTHS else "15 минут")
        self.len_menu = ctk.CTkOptionMenu(o, variable=self.len_var, values=FILE_LENGTHS, height=32,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_: (self._remember(), self.reestimate()))
        self.len_menu.pack(fill="x", pady=(6, 0))

        T.section(o, "Формат").pack(fill="x", pady=(16, 0))
        self.fmt_var = ctk.StringVar(value=self.cfg.get("fmt", DEFAULT_FORMAT)
                                     if self.cfg.get("fmt") in FORMATS else DEFAULT_FORMAT)
        self.fmt_menu = ctk.CTkOptionMenu(o, variable=self.fmt_var, values=list(FORMATS), height=32,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_: (self._remember(), self.show_estimate()))
        self.fmt_menu.pack(fill="x", pady=(6, 0))

        ctk.CTkFrame(o, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=16)
        T.section(o, "Движок").pack(fill="x")
        self.profile_chip = T.chip(o, "", "acc")
        self.profile_chip.pack(anchor="w", pady=(6, 2))
        self.accel_hint = T.label(o, "", 12, color=T.MUTED, anchor="w", justify="left",
                                  wraplength=290)
        self.accel_hint.pack(fill="x")
        T.label(o, "Движок и паузы меняются в Настройках.", 12, color=T.MUTED,
                anchor="w").pack(fill="x")

        ctk.CTkFrame(o, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=16)
        self.preview_btn = T.button(o, "Проверить голос (1 мин)", "play", height=36, size=14,
                                    command=self.do_preview)
        self.preview_btn.pack(fill="x")
        T.label(o, "Минута из середины книги - послушать голос до многочасовой озвучки.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=290).pack(fill="x", pady=(6, 0))
        self.readtext_btn = T.button(o, "Как будет прочитано", "file-text", kind="ghost",
                                     height=30, size=13, command=self.show_read_text)
        self.readtext_btn.pack(fill="x", pady=(8, 0))
        self._setting_widgets = [self.voice_menu, self.len_menu, self.fmt_menu]

    # ================= страница «Голоса» =================
    def _build_voices_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=28, pady=(22, 14))
        T.label(head, "Голоса", 22, bold=True, anchor="w").pack(fill="x")
        T.label(head, "Клон делается из любой записи диктора - например, его аудиокниги. "
                      "Готовые голоса Silero работают сразу и очень быстро.",
                13, color=T.MUTED, anchor="w").pack(fill="x", pady=(2, 0))

        lst = T.card(page)
        lst.grid(row=1, column=0, sticky="nsew", padx=(28, 8), pady=(0, 20))
        hdr = ctk.CTkFrame(lst, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 8))
        T.label(hdr, "Все голоса", 15, bold=True, anchor="w").pack(side="left")
        self.add_voice_btn = T.button(hdr, "Добавить образец…", "plus", kind="primary", height=30,
                                      size=13, width=170, command=self.load_sample)
        self.add_voice_btn.pack(side="right")
        T.button(hdr, "Папка", "folder-open", kind="ghost", height=30, size=13, width=100,
                 command=lambda: _open_path(default_voices_dir())).pack(side="right", padx=6)
        ctk.CTkFrame(lst, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x")
        self.voice_rows = ctk.CTkScrollableFrame(lst, fg_color="transparent", corner_radius=0)
        self.voice_rows.pack(fill="both", expand=True, padx=1, pady=(0, 6))

        fx = T.card(page)
        fx.grid(row=1, column=1, sticky="nsew", padx=(8, 28), pady=(0, 20))
        self.fx_box = ctk.CTkFrame(fx, fg_color="transparent")
        self.fx_box.pack(fill="both", expand=True, padx=18, pady=16)
        ctk.CTkFrame(self.fx_box, width=300, height=0, fg_color="transparent").pack()
        T.section(self.fx_box, "Звучание голоса").pack(fill="x")
        self.fx_title = T.label(self.fx_box, "Выберите голос слева", 16, bold=True, anchor="w")
        self.fx_title.pack(fill="x", pady=(6, 0))
        self.fx_src = T.label(self.fx_box, "", 12, color=T.MUTED, anchor="w", justify="left",
                              wraplength=300)
        self.fx_src.pack(fill="x", pady=(0, 6))
        T.label(self.fx_box, "Поправки запоминаются для голоса и применяются при каждой "
                             "озвучке. Интонацию так не поменять - она приходит из образца.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=300).pack(fill="x", pady=(0, 10))
        for title, key, lo, hi, unit in (("Темп", "speed", 0.8, 1.25, "×"),
                                         ("Высота", "pitch", -4, 4, " полутона"),
                                         ("Бас", "bass", -6, 6, " дБ"),
                                         ("Яркость", "brightness", -6, 6, " дБ")):
            line = ctk.CTkFrame(self.fx_box, fg_color="transparent")
            line.pack(fill="x", pady=4)
            T.label(line, title, 13, anchor="w", width=70).pack(side="left")
            val = T.label(line, "", 12, color=T.MUTED, width=90, anchor="e")
            val.pack(side="right")
            steps = 18 if key == "speed" else int((hi - lo) * 4)
            sl = ctk.CTkSlider(line, from_=lo, to=hi, number_of_steps=steps,
                               button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
                               progress_color=T.ACCENT, fg_color=T.SURFACE2)
            sl.pack(side="left", fill="x", expand=True, padx=8)
            self._fx_vars[key] = (sl, val, unit)
            sl.configure(command=lambda _v, k=key: self._fx_slider(k))
        btns = ctk.CTkFrame(self.fx_box, fg_color="transparent")
        btns.pack(fill="x", pady=(14, 0))
        self.fx_play_btn = T.button(btns, "Прослушать", "play", kind="primary", height=34, width=140,
                                    command=self._fx_play)
        self.fx_play_btn.pack(side="left")
        T.button(btns, "Сбросить", "refresh-cw", kind="ghost", height=34, width=110,
                 command=self._fx_reset).pack(side="left", padx=6)
        self.fx_save_btn = T.button(btns, "Сохранить", "save", height=34, width=120,
                                    command=self._fx_save)
        self.fx_save_btn.pack(side="right")
        self.fx_note = T.label(self.fx_box, "", 12, color=T.OK, anchor="w")
        self.fx_note.pack(fill="x", pady=(8, 0))
        self._fx_enable(False)
        return page

    # ================= страница «Настройки» =================
    def _build_settings_page(self):
        # Обычная рамка снаружи, прокрутка внутри: у CTkScrollableFrame
        # grid_remove прячет только внутреннюю часть, и страница оставалась
        # видна поверх других (поймал тест окна)
        outer = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page = ctk.CTkScrollableFrame(outer, fg_color="transparent", corner_radius=0)
        page.pack(fill="both", expand=True)
        self.settings_scroll = page
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.pack(fill="x", padx=28, pady=(22, 14))
        T.label(head, "Настройки", 22, bold=True, anchor="w").pack(fill="x")

        c = T.card(page)
        c.pack(fill="x", padx=28, pady=(0, 14))
        b = ctk.CTkFrame(c, fg_color="transparent")
        b.pack(fill="x", padx=18, pady=16)
        T.section(b, "Движок").pack(fill="x")
        T.label(b, "Выбран слепым прослушиванием главы: лучший - «Основной». Готовые "
                   "голоса Silero включаются выбором голоса.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", pady=(2, 8))
        self.profile_var = ctk.StringVar(value=profiles.get(self.cfg.get("profile")).key)
        self._profile_radios = []
        for p in profiles.PROFILES.values():
            line = ctk.CTkFrame(b, fg_color="transparent")
            line.pack(fill="x", pady=(8, 0))
            r = ctk.CTkRadioButton(line, text=p.title, value=p.key, variable=self.profile_var,
                                   font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                                   border_color=T.BORDER_STRONG, text_color=T.TEXT,
                                   command=self._on_profile)
            r.pack(side="left")
            # «Послушать»: та же фраза, что в «Голосах», выбранным голосом и этим движком
            pb = T.button(line, "Послушать", "play", kind="ghost", height=26, size=12,
                          width=110, command=lambda k=p.key: self._profile_listen(k))
            pb.pack(side="left", padx=(10, 0))
            self._prof_btns[p.key] = pb
            self._profile_radios.append(r)
            T.label(b, p.hint, 12, color=T.MUTED, anchor="w", justify="left",
                    wraplength=640).pack(fill="x", padx=(30, 0))

        cp = T.card(page)
        cp.pack(fill="x", padx=28, pady=(0, 14))
        bp = ctk.CTkFrame(cp, fg_color="transparent")
        bp.pack(fill="x", padx=18, pady=16)
        T.section(bp, "Паузы").pack(fill="x")
        T.label(bp, "Тишина после предложения, после абзаца и между репликами диалога "
                    "(смена говорящего - поэтому длиннее).", 12, color=T.MUTED, anchor="w",
                justify="left", wraplength=640).pack(fill="x", pady=(2, 6))
        saved = self.cfg.get("pauses", {})
        for key, title, dflt, lo, hi in PAUSES:
            line = ctk.CTkFrame(bp, fg_color="transparent")
            line.pack(fill="x", pady=3)
            T.label(line, title, 13, anchor="w", width=170).pack(side="left")
            val = T.label(line, "", 12, color=T.MUTED, width=70, anchor="e")
            val.pack(side="right")
            sl = ctk.CTkSlider(line, from_=lo, to=hi, number_of_steps=int(round((hi - lo) / 0.05)),
                               button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
                               progress_color=T.ACCENT, fg_color=T.SURFACE2,
                               command=lambda _v, k=key: self._on_pause(k))
            sl.pack(side="left", fill="x", expand=True, padx=8)
            try:
                sl.set(float(saved.get(key, dflt)))
            except (TypeError, ValueError):
                sl.set(dflt)
            self._pause_vars[key] = (sl, val)
            self._on_pause(key, save=False)
        T.button(bp, "Как было", "refresh-cw", kind="ghost", height=28, size=13, width=120,
                 command=self._pauses_reset).pack(anchor="w", pady=(6, 0))

        self.prof_note = T.label(b, "", 12, color=T.MUTED, anchor="w")
        self.prof_note.pack(fill="x", pady=(8, 0))

        cq = T.card(page)
        cq.pack(fill="x", padx=28, pady=(0, 14))
        bq = ctk.CTkFrame(cq, fg_color="transparent")
        bq.pack(fill="x", padx=18, pady=16)
        T.section(bq, "Качество / скорость").pack(fill="x")
        T.label(bq, "Для клона (основной и ESpeech). Модель и 12 шагов те же; меняется, на "
                    "скольких шагах модель сверяется с текстом дважды.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", pady=(2, 6))
        qk = self.cfg.get("quality")
        self.quality_var = ctk.StringVar(value=qk if qk in profiles.QUALITY else profiles.DEFAULT_QUALITY)
        self._quality_radios = []
        for q in profiles.QUALITY.values():
            r = ctk.CTkRadioButton(bq, text=q.title, value=q.key, variable=self.quality_var,
                                   font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                                   border_color=T.BORDER_STRONG, text_color=T.TEXT,
                                   command=self._on_quality)
            r.pack(anchor="w", pady=(8, 0))
            self._quality_radios.append(r)
            T.label(bq, q.hint, 12, color=T.MUTED, anchor="w", justify="left",
                    wraplength=640).pack(fill="x", padx=(30, 0))

        cs = T.card(page)
        cs.pack(fill="x", padx=28, pady=(0, 14))
        bs = ctk.CTkFrame(cs, fg_color="transparent")
        bs.pack(fill="x", padx=18, pady=16)
        T.section(bs, "Проверка").pack(fill="x")
        self.sound_var = ctk.BooleanVar(value=bool(self.cfg.get("sound_check", True)))
        self.sound_chk = ctk.CTkCheckBox(
            bs, text="Проверка слов по звуку", variable=self.sound_var,
            font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
            border_color=T.BORDER_STRONG, text_color=T.TEXT, command=self._on_strict)
        self.sound_chk.pack(anchor="w", pady=(8, 0))
        T.label(bs, "Текст книги укладывается на звук моделью без языковой догадки: "
                    "съеденное, смазанное или заменённое слово («ногой» -> «новой») видно "
                    "по звуку, и кусок пересчитывается с другим зерном (до двух раз); "
                    "берётся вариант с лучшим худшим словом. Края кусков проверяются всегда.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", padx=(30, 0))
        # строгая проверка 3.2 (текст Whisper против текста книги) убрана из окна
        # в 3.3: на слух выигрыша не дала, а время +50 %; проверка по звуку лучше
        # на размеченном наборе (9 из 10 мест против 7 из 10)
        self.strict_var = ctk.BooleanVar(value=False)

        cr = T.card(page)
        cr.pack(fill="x", padx=28, pady=(0, 14))
        br = ctk.CTkFrame(cr, fg_color="transparent")
        br.pack(fill="x", padx=18, pady=16)
        T.section(br, "Прочитать вслух").pack(fill="x")
        self.say_clones_var = ctk.BooleanVar(value=bool(self.cfg.get("say_clones", False)))
        ctk.CTkCheckBox(br, text="Показывать клон-голоса в «Прочитать вслух»",
                        variable=self.say_clones_var, font=T.font(14), fg_color=T.ACCENT,
                        hover_color=T.ACCENT_HOVER, border_color=T.BORDER_STRONG,
                        text_color=T.TEXT, command=self._on_say_clones).pack(anchor="w", pady=(8, 0))
        T.label(br, "Клон в этой вкладке начинает говорить только через 10-50 с; быстрые "
                    "голоса - сразу. Для книги клон - лучший выбор.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", padx=(30, 0))

        csv = T.card(page)
        csv.pack(fill="x", padx=28, pady=(0, 14))
        bv = ctk.CTkFrame(csv, fg_color="transparent")
        bv.pack(fill="x", padx=18, pady=16)
        T.section(bv, "Озвучка для других программ").pack(fill="x")
        from core.speak_server import DEFAULT_PORT, DEFAULT_IDLE_MIN
        self.srv_var = ctk.BooleanVar(value=bool(self.cfg.get("speak_server", False)))
        self.srv_chk = ctk.CTkCheckBox(
            bv, text="Включить сервис /speak", variable=self.srv_var, font=T.font(14),
            fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER, border_color=T.BORDER_STRONG,
            text_color=T.TEXT, command=self._srv_toggle)
        self.srv_chk.pack(anchor="w", pady=(8, 0))
        T.label(bv, "Другие программы на этом компьютере (Legal Knowledge Desk, помощник по "
                    "настольной игре) просят Herald прочитать текст: POST /speak - и звук идёт "
                    "по мере готовности. Только 127.0.0.1, без интернета. Как вызвать - в README.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", padx=(30, 0))
        line = ctk.CTkFrame(bv, fg_color="transparent")
        line.pack(fill="x", padx=(30, 0), pady=(8, 0))
        T.label(line, "Порт", 13, anchor="w").pack(side="left")
        self.srv_port = ctk.CTkEntry(line, width=80, height=28, font=T.font(13))
        self.srv_port.insert(0, str(self.cfg.get("speak_port", DEFAULT_PORT)))
        self.srv_port.pack(side="left", padx=(8, 18))
        self.srv_port.bind("<Return>", lambda e: self._srv_apply())
        self.srv_port.bind("<FocusOut>", lambda e: self._srv_apply())
        T.label(line, "Выгружать модели после простоя", 13, anchor="w").pack(side="left")
        idle_vals = ["5 минут", "10 минут", "30 минут", "60 минут"]
        cur = f"{int(self.cfg.get('speak_idle', DEFAULT_IDLE_MIN))} минут"
        self.srv_idle = ctk.CTkOptionMenu(line, values=idle_vals, width=110, height=28,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_: self._srv_apply())
        self.srv_idle.set(cur if cur in idle_vals else "10 минут")
        self.srv_idle.pack(side="left", padx=(8, 0))
        self.srv_note = T.label(bv, "", 12, color=T.MUTED, anchor="w")
        self.srv_note.pack(fill="x", padx=(30, 0), pady=(6, 0))

        c2 = T.card(page)
        c2.pack(fill="x", padx=28, pady=(0, 14))
        b2 = ctk.CTkFrame(c2, fg_color="transparent")
        b2.pack(fill="x", padx=18, pady=16)
        T.section(b2, "Для опытных").pack(fill="x")
        self.tech_label = T.label(b2, "", 12, color=T.MUTED, anchor="w", justify="left",
                                  wraplength=680)
        self.tech_label.pack(fill="x", pady=(6, 8))
        row = ctk.CTkFrame(b2, fg_color="transparent")
        row.pack(fill="x")
        # Ускорение NVIDIA (Windows): видно, только если есть видеокарта, а
        # считается на процессоре - torch с CUDA докачивается отдельно (~2,5 ГБ)
        self.nv_box = ctk.CTkFrame(b2, fg_color="transparent")
        self.nv_note = T.label(self.nv_box, "Найдена видеокарта NVIDIA. Чтобы озвучка шла на ней, "
                                            "нужна сборка torch с CUDA (около 2,5 ГБ, скачивается "
                                            "отдельно, чтобы не раздувать установщик).",
                               12, color=T.MUTED, anchor="w", justify="left", wraplength=640)
        self.nv_note.pack(fill="x")
        self.nv_btn = T.button(self.nv_box, "Скачать ускорение NVIDIA", "upload", height=30,
                               size=13, width=240, command=self._nvidia_download)
        self.nv_btn.pack(anchor="w", pady=(6, 8))
        T.button(row, "Открыть журнал", "circle-alert", height=30, size=13, width=170,
                 command=lambda: _open_path(LOG_PATH) if os.path.exists(LOG_PATH)
                 else messagebox.showinfo("Журнал", "Журнал пока пуст - проблем не было.")
                 ).pack(side="left")
        self.showlog_var = ctk.BooleanVar(value=bool(self.cfg.get("show_log", False)))
        ctk.CTkCheckBox(row, text="Показывать технический журнал на главной",
                        variable=self.showlog_var, font=T.font(13), fg_color=T.ACCENT,
                        hover_color=T.ACCENT_HOVER, border_color=T.BORDER_STRONG,
                        text_color=T.TEXT, command=self._on_showlog).pack(side="left", padx=(14, 0))

        c3 = T.card(page)
        c3.pack(fill="x", padx=28, pady=(0, 22))
        box = ctk.CTkTextbox(c3, height=320, wrap="word", font=T.font(12), border_width=0,
                             fg_color=T.SURFACE)
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", ABOUT.format(v=VERSION, tag=APP_TAGLINE) if "{v}" in ABOUT else ABOUT)
        box.configure(state="disabled")
        self._update_tech()
        return outer

    def _update_tech(self):
        p = self.profile()
        kw = ", ".join(f"{k}={v}" for k, v in p.kwargs.items()) or "-"
        got = self.cfg.get("rtf", {}).get(self._rtf_key())
        measured = (f"замерено при последней озвучке: {got:.2f}" if got
                    else "замера ещё не было - оценка по стенду")
        self.tech_label.configure(text=(
            f"«{p.title}»: движок {p.engine}, параметры {kw}. Счёт на секунду звука "
            f"(RTF) ~{p.rtf:.2f} с на Apple M4, {measured}. Шаги - число уточнений "
            "картинки звука (NFE), расписание EPSS - прореженные шаги из работы "
            "Fast F5-TTS. Готовые куски "
            "запоминаются (кэш синтеза), поэтому смена пауз или формата пересобирает "
            "книгу за минуты. Журнал: " + LOG_PATH))

    def _pause(self, key: str) -> float:
        sl, _v = self._pause_vars[key]
        return round(sl.get() / 0.05) * 0.05

    def _on_pause(self, key: str, save: bool = True):
        _sl, val = self._pause_vars[key]
        val.configure(text=f"{self._pause(key):.2f} с".replace(".", ","))
        if save:
            self.cfg["pauses"] = {k: round(self._pause(k), 2) for k in self._pause_vars}
            save_config(self.cfg)
            self._check_resume()

    def _on_strict(self):
        self.cfg["strict_words"] = bool(self.strict_var.get())
        self.cfg["sound_check"] = bool(self.sound_var.get())
        save_config(self.cfg)
        self._check_resume()
        self.show_estimate()

    def _pauses_reset(self):
        for key, _t, dflt, _lo, _hi in PAUSES:
            self._pause_vars[key][0].set(dflt)
            self._on_pause(key, save=False)
        self.cfg.pop("pauses", None)
        save_config(self.cfg)

    def _rtf(self) -> float:
        """RTF для прогноза: замеренный на этом компьютере, если был, иначе по стенду."""
        v = self._voice_obj()
        if v is not None and v.ready:
            return profiles.SILERO.rtf
        p = self.profile()
        # сначала - полное ожидание прошлых книг этим движком (со сборкой и
        # пересчётами), потом - замер счёта по частям, потом прикидка профиля
        key = self._rtf_key()
        rw = self.cfg.get("rtf_wall", {})
        # этот голос в этом режиме -> любой голос в этом режиме -> счёт по частям
        got = (rw.get(f"{key}|{v.id}" if v else key) or rw.get(f"{key}|*")
               or self.cfg.get("rtf", {}).get(key))
        if got:
            return float(got)
        return (profiles.rtf_for(p, self._accel, self._quality())
                * (STRICT_COST if self.strict_var.get() else 1.0)
                * (SOUND_COST if self.sound_var.get() else 1.0))

    def _show_accel(self, accel: str):
        """Чем считается озвучка; на процессоре - честное предупреждение."""
        self._accel = accel
        try:
            from core import cuda_pack
            show_nv = (osutil.IS_WIN and accel == "cpu" and not cuda_pack.ready()
                       and osutil.has_nvidia())
        except Exception:
            show_nv = False
        if show_nv and not self.nv_box.winfo_manager():
            self.nv_box.pack(fill="x", before=self.tech_label)
        if accel == "cpu":
            self.accel_hint.configure(
                text="Считается на процессоре: час звука - около суток счёта, книга - "
                     "несколько суток. Быстро на процессоре работают готовые голоса "
                     "Silero (страница «Голоса»).",
                text_color=T.WARN)
        else:
            self.accel_hint.configure(text="Считается: " + profiles.ACCEL_TITLE.get(accel, accel),
                                      text_color=T.MUTED)
        self.show_estimate()

    def _nvidia_download(self):
        from core import cuda_pack
        self.nv_btn.configure(state="disabled", text="Скачиваю…")

        def prog(got, total):
            pct = int(got * 100 / total) if total else 0
            self.ui(lambda: self.nv_btn.configure(text=f"Ускорение NVIDIA: {pct}%"))

        def work():
            try:
                cuda_pack.download(prog)
                self.ui(lambda: (self.nv_btn.configure(text="Готово"),
                                 self.nv_note.configure(text="Ускорение скачано. Перезапустите "
                                                             "Herald - озвучка пойдёт на видеокарте.",
                                                        text_color=T.OK)))
            except Exception as e:
                log_problem("ускорение NVIDIA не скачалось", e, show=False)
                self.ui(lambda: (self.nv_btn.configure(state="normal",
                                                       text="Скачать ускорение NVIDIA"),
                                 self.nv_note.configure(text=f"Не скачалось: {e}",
                                                        text_color=T.BAD)))
        threading.Thread(target=work, daemon=True).start()

    def _rtf_key(self) -> str:
        """Замер RTF - на профиль И число шагов: при смене шагов старый не годится."""
        p = self.profile()
        strict = ("s" if self.strict_var.get() else "n") + ("z" if self.sound_var.get() else "")
        # 3.6: и версия, и режим - замеры прежних версий и других режимов не годятся
        # (прогноз Ященко в 3.5 шёл по замерам до ускорения: ×1,8 вместо ×1,4)
        return (f"{VERSION}:{p.key}:{p.kwargs.get('steps', '')}:{self._quality()}:"
                f"{self._accel}:{strict}")

    def _got_rtf(self, rtf: float):
        """Настоящий RTF от конвейера: показать и запомнить для прогноза."""
        self._live_rtf = rtf
        v = self._voice_obj()
        if v is not None and not v.ready:
            key = self._rtf_key()
            old = self.cfg.setdefault("rtf", {}).get(key)
            # сглаживаем: одна часть с кэшем или переозвучками не должна дёргать прогноз
            self.cfg["rtf"][key] = round(rtf if not old else 0.7 * old + 0.3 * rtf, 3)
            save_config(self.cfg)
            self._update_tech()

    def _learn_book(self, dur: float, took: float):
        """Книга досчитана: запомнить для прогноза настоящее ожидание на секунду
        звука (со всеми проверками, пересчётами и сборкой) и поправку длины звука
        этого голоса (3.5: прогноз 3:01 против факта 3:22)."""
        v = self._voice_obj()
        if v is None or dur < 20 or not self.est or not self.est.audio_sec:
            return
        rate = (load_fx(default_voices_dir(), v.id).speed or 1.0) if not v.ready else 1.0
        k = dur / (self.est.audio_sec / rate)
        ak = self.cfg.setdefault("audio_k", {})
        ak[v.id] = round(k if v.id not in ak else 0.3 * ak[v.id] + 0.7 * k, 3)
        if not v.ready:
            key = self._rtf_key()
            rw = self.cfg.setdefault("rtf_wall", {})
            w = took / dur
            for k_ in (f"{key}|{v.id}", f"{key}|*"):     # голос и режим целиком
                rw[k_] = round(w if k_ not in rw else 0.3 * rw[k_] + 0.7 * w, 3)
        save_config(self.cfg)
        self.show_estimate()

    def profile(self) -> profiles.Profile:
        return profiles.get(self.profile_var.get())

    def _quality(self) -> str:
        return self.quality_var.get() if hasattr(self, "quality_var") else profiles.DEFAULT_QUALITY

    def _prof_kwargs(self, p: profiles.Profile | None = None) -> dict:
        """Параметры движка профиля с режимом «Качество / скорость» (3.6)."""
        return profiles.engine_kwargs(p or self.profile(), self._quality())

    def _on_say_clones(self):
        self.cfg["say_clones"] = bool(self.say_clones_var.get())
        save_config(self.cfg)
        if hasattr(self, "say_mode"):
            self._say_on_kind(self.say_mode.get() if self._say_clones() else "Быстрый",
                              remember=False)

    def _on_quality(self):
        self.cfg["quality"] = self.quality_var.get()
        save_config(self.cfg)
        self._srv_sync()
        self._update_tech()
        self.show_estimate()
        if hasattr(self, "say_forecast"):
            self._say_forecast()

    def _on_profile(self):
        self.cfg["profile"] = self.profile_var.get()
        save_config(self.cfg)
        self._srv_sync()
        self._update_tech()
        self.refresh_voices()
        self.show_estimate()

    def _remember(self):
        self.cfg.update(len=self.len_var.get(), fmt=self.fmt_var.get(),
                        voice=self.current_voice())
        save_config(self.cfg)

    # ================= голоса =================
    def _all_voices(self) -> list[_V]:
        out = []
        try:
            for v in get_engine(clone_engine()).list_voices():
                out.append(_V(v.id, v.id, v.language, False))
        except Exception as e:
            log_problem("не удалось прочитать список голосов", e)
        out += [_V(v.id, v.title, v.language, True, "silero", v.gender) for v in SILERO_VOICES]
        # быстрые голоса Vosk / Piper - только отмеченные Андреем (3.6, core/engines/fast)
        try:
            from core.engines import fast
            out += [_V(v.id, v.title, v.language, True, fast.engine_of(v.id), v.gender)
                    for v in fast.enabled_voices()]
        except Exception as e:
            log_problem("быстрые голоса не прочитались", e, show=False)
        return out

    def refresh_voices(self):
        # все голоса в одном списке: сначала клоны, потом готовые Silero
        self._voices = sorted(self._all_voices(), key=lambda v: v.ready)
        lang = getattr(self.est, "language", None)
        if lang:
            self._voices.sort(key=lambda v: (v.language != lang, v.ready))
        labels = [v.label() for v in self._voices] or [""]
        self.voice_menu.configure(values=labels)
        want = self.cfg.get("voice")
        fits = [v for v in self._voices if not lang or v.language == lang] or self._voices
        cur = next((v.label() for v in fits if v.id == want), None)
        now = next((v for v in self._voices if v.label() == self.voice_var.get()), None)
        # голос другого языка книге не подходит: берём сохранённый подходящий,
        # иначе первый подходящий
        if now is None or (lang and now.language != lang and fits[0].language == lang):
            self.voice_var.set(cur or (fits[0].label() if fits else labels[0]))
        self._on_voice(remember=False)
        if hasattr(self, "say_mode"):              # вкладка «Прочитать вслух» - те же голоса
            self._say_on_kind(self.say_mode.get(), remember=False)

    def current_voice(self) -> str:
        label = self.voice_var.get()
        for v in self._voices:
            if v.label() == label:
                return v.id
        return label.split(" · ")[0]

    def _voice_obj(self, vid: str | None = None) -> _V | None:
        vid = vid or self.current_voice()
        return next((v for v in self._all_voices() if v.id == vid), None)

    def _on_voice(self, remember: bool = True):
        v = self._voice_obj()
        if v is None:
            self.voice_hint.configure(text="Голосов пока нет - добавьте на странице «Голоса».")
        else:
            lang = getattr(self.est, "language", None)
            names = {"ru": "русском", "en": "английском"}
            warn = (f" Книга на {names.get(lang, lang)}, а голос - на "
                    f"{names.get(v.language, v.language)}: выйдет с акцентом."
                    if lang and lang != v.language else "")
            kind = "готовый голос, без клонирования" if v.ready else "клон по образцу"
            self.voice_hint.configure(text=kind + "." + warn,
                                      text_color=T.WARN if warn else T.MUTED)
        if self.profile_chip:
            T.set_chip(self.profile_chip, profiles.SILERO.title if v is not None and v.ready
                       else self.profile().title, "acc")
        if remember:
            self._remember()
        self._srv_sync()
        self.show_estimate()

    def _refresh_voice_rows(self):
        """Список голосов: вся строка - кнопка. Наведение подсвечивает, щелчок
        выбирает голос для озвучки и открывает справа его звучание."""
        for w in self.voice_rows.winfo_children():
            w.destroy()
        self._voice_row_widgets = {}
        allv = self._all_voices()
        groups = [("Клоны (ваши дикторы)", [v for v in allv if not v.ready]),
                  ("Готовые Silero (быстро, без клонирования; лицензия некоммерческая)",
                   [v for v in allv if v.ready])]
        fxdir = default_voices_dir()
        chosen = self.current_voice()
        for gtitle, vs in groups:
            T.section(self.voice_rows, gtitle).pack(fill="x", padx=14, pady=(12, 4))
            if not vs:
                T.label(self.voice_rows, "пока ни одного", 12, color=T.MUTED, anchor="w").pack(
                    fill="x", padx=14)
            for v in vs:
                row = ctk.CTkFrame(self.voice_rows, fg_color="transparent", corner_radius=8,
                                   cursor=osutil.HAND)
                row.pack(fill="x", padx=8, pady=1)
                row.grid_columnconfigure(1, weight=1)
                ic = ctk.CTkLabel(row, text="", image=T.icon("mic", T.MUTED, 18))
                ic.grid(row=0, column=0, rowspan=2, padx=(10, 10), pady=(8, 0), sticky="n")
                name = T.label(row, v.title, 14, bold=True, anchor="w")
                name.grid(row=0, column=1, sticky="ew", pady=(6, 0))
                meta = [LANG_MARK.get(v.language, v.language),
                        "готовый" if v.ready else "клон"]
                if not v.ready and not load_fx(fxdir, v.id).is_neutral():
                    meta.append("звучание настроено")
                if v.id == chosen:
                    meta.append("выбран для озвучки")
                sub = T.label(row, "  ·  ".join(meta), 12, color=T.MUTED, anchor="w")
                sub.grid(row=1, column=1, sticky="ew", pady=(0, 6))
                mark = ctk.CTkLabel(row, text="", width=24, image=T.icon(
                    "check" if v.id == chosen else "settings",
                    T.ACCENT if v.id == chosen else T.FAINT, 16))
                mark.grid(row=0, column=2, rowspan=2, padx=(4, 12))
                parts = (row, ic, name, sub, mark)
                for w in parts:
                    w.bind("<Button-1>", lambda e, vid=v.id: self.pick_voice(vid))
                    w.bind("<Enter>", lambda e, vid=v.id: self._voice_hover(vid, True))
                    w.bind("<Leave>", lambda e, vid=v.id: self._voice_hover(vid, False))
                self._voice_row_widgets[v.id] = row
                self._voice_hover(v.id, False)

    def _voice_row_color(self, vid: str, hover: bool):
        if vid == self._fx_voice:
            return T.ACCENT_SOFT
        return T.SURFACE2 if hover else "transparent"

    def _voice_hover(self, vid: str, on: bool):
        row = self._voice_row_widgets.get(vid)
        if row is None or not row.winfo_exists():
            return
        if not on:
            # уход с подписи на саму строку - тоже <Leave>; гасим, только если
            # указатель правда вышел за строку
            x, y = self.winfo_pointerxy()
            w = self.winfo_containing(x, y)
            while w is not None:
                if w is row:
                    return
                w = getattr(w, "master", None)
        row.configure(fg_color=self._voice_row_color(vid, on))

    def pick_voice(self, vid: str):
        """Щелчок по голосу: он становится голосом озвучки, справа - его звучание."""
        v = next((x for x in self._voices if x.id == vid), None)
        if v is not None and not self.running:
            self.voice_var.set(v.label())
            self._on_voice()
        self.select_fx_voice(vid)

    def select_fx_voice(self, vid: str):
        """Открыть настройки звучания голоса справа."""
        self._fx_voice = vid
        self._fx_raw = None
        v = self._voice_obj(vid)
        self.fx_title.configure(text=v.title if v else vid)
        src_txt = "готовый голос Silero" if v and v.ready else ""
        sp = os.path.join(default_voices_dir(), vid + ".src.json")
        if os.path.exists(sp):
            try:
                src_txt = "образец: " + os.path.basename(
                    json.load(open(sp, encoding="utf-8"))["source"])
            except (OSError, ValueError, KeyError):
                pass            # нет сведений об образце - просто не пишем
        self.fx_src.configure(text=src_txt)
        fx = load_fx(default_voices_dir(), vid)
        for k, (sl, _l, _u) in self._fx_vars.items():
            sl.set(getattr(fx, k))
            self._fx_changed(k)
        self._fx_enable(True)
        self.fx_note.configure(text="")
        self._refresh_voice_rows()

    def _fx_enable(self, on: bool):
        st = "normal" if on else "disabled"
        for sl, _l, _u in self._fx_vars.values():
            sl.configure(state=st)
        for b in (self.fx_play_btn, self.fx_save_btn):
            b.configure(state=st)

    def _fx_current(self) -> VoiceFX:
        return VoiceFX(**{k: round(sl.get(), 2 if k == "speed" else 1)
                          for k, (sl, _l, _u) in self._fx_vars.items()})

    def _fx_slider(self, key: str):
        self._fx_changed(key)
        if key == "speed":
            self._fx_raw = None          # темп задаётся при синтезе - фразу пересчитаем

    def _fx_changed(self, key: str):
        sl, lbl, unit = self._fx_vars[key]
        if key == "speed":
            v = round(sl.get(), 2)
            lbl.configure(text="как есть" if abs(v - 1) < 0.01 else f"{v:.2f}{unit}")
        else:
            v = round(sl.get(), 1)
            lbl.configure(text="нет" if v == 0 else f"{v:+.1f}{unit}")

    def _fx_reset(self):
        for k, (sl, _l, _u) in self._fx_vars.items():
            sl.set(1.0 if k == "speed" else 0)
            self._fx_changed(k)

    def _fx_play(self):
        """«Прослушать»: фраза потоком. Сырой звук запоминается - дальше ползунки
        (кроме темпа) меняют только обработку, и повтор звучит мгновенно."""
        if self._live_kind == "fx":
            self._live_stop()
            return
        vid = self._fx_voice
        if not vid:
            return
        v = self._voice_obj(vid)
        fx = self._fx_current()
        raw0 = self._fx_raw
        if raw0 and os.path.exists(raw0):
            self._fx_play_raw(raw0, fx)
            return
        lang = v.language if v else "ru"
        if v and v.ready:
            engine, kw = v.engine, {}
            text = self.FX_PHRASE.get(v.language, self.FX_PHRASE["ru"])
        else:
            kw = dict(self._prof_kwargs(), speed=fx.speed)
            if lang != "ru":
                kw.pop("variant", None)
            engine, text = clone_engine(), self.FX_PHRASE.get(lang, self.FX_PHRASE["ru"])

        def done(audio, sr):
            if audio is None or not len(audio):
                return
            import tempfile
            import soundfile as sf
            raw = os.path.join(tempfile.mkdtemp(prefix="herald_fx_"), "raw.wav")
            sf.write(raw, audio, sr)
            if self._fx_voice == vid:
                self._fx_raw = raw
        self._speak_job("fx", text, engine, kw, vid, prepared=True, fx_chain=fx.filter_chain(),
                        note=self.fx_note, on_done=done)

    def _fx_play_raw(self, raw: str, fx: VoiceFX):
        """Повтор с другими ползунками: сырой звук уже есть - только обработка."""
        self._live_stop()
        self._live_kind = "fx"
        self._live_buttons()

        def work():
            try:
                out = os.path.join(os.path.dirname(raw), "play.wav")
                chain = ",".join(x for x in (HD_FILTER, fx.filter_chain()) if x)
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                "-i", raw, "-af", chain, out], check=True)
                fxp = Player()
                if fxp.load(out):
                    self._fx_player = fxp
                    fxp.play()
                    while fxp.playing and self._fx_player is fxp:
                        time.sleep(0.1)
                    fxp.stop()
                else:
                    _open_path(out)
            except Exception as e:
                log_problem("прослушивание голоса не удалось", e, show=False)
                self.ui(lambda: messagebox.showerror("Ошибка", str(e)))
            finally:
                def end():
                    if self._live_kind == "fx" and self._live is None:
                        self._live_kind = ""
                    self._fx_player = None
                    self._live_buttons()
                self.ui(end)
        threading.Thread(target=work, daemon=True).start()

    # ================= звук потоком (core/stream) =================
    def _live_buttons(self):
        """Кнопка того, что сейчас звучит, - «Стоп»; остальные - как обычно."""
        k = self._live_kind
        for btn, kind, title, icon, style in (
                [(self.fx_play_btn, "fx", "Прослушать", "play", "primary")] +
                [(b, "prof:" + key, "Послушать", "play", "ghost")
                 for key, b in self._prof_btns.items()]):
            if k == kind:
                btn.configure(text="Стоп", state="normal")
                T.restyle(btn, "danger", "square")
            else:
                btn.configure(text=title)
                T.restyle(btn, style, icon)
                ok = not self.running and not (btn is self.fx_play_btn and not self._fx_voice)
                btn.configure(state="normal" if ok else "disabled")
        if hasattr(self, "say_pause_btn"):
            self._say_buttons()

    def _live_stop(self):
        live, self._live = self._live, None
        if live is not None:
            live.stop()
        fxp, self._fx_player = self._fx_player, None
        if fxp is not None:
            fxp.stop()
        if self._live_kind != "preview":
            self._live_kind = ""
        self._live_buttons()

    def _live_new(self, kind: str, engine: str, kw: dict, voice: str, sound_check: bool,
                  show, fx_chain: str | None = None, gate: bool = True):
        """Новый поток (прежний останавливается). show(текст) - куда писать статус."""
        self._live_stop()
        prior = (profiles.SILERO.rtf if not _clone_engine(engine)
                 else profiles.rtf_for(self.profile(), self._accel)
                 * (SOUND_COST if sound_check else 1.0))
        live = S.Live(S.default_sink(),
                      S.make_speed(engine, kw, voice, sound_check, prior, self._accel),
                      fx_chain=fx_chain, gate=gate)

        def status(st):
            def apply():
                if self._live is not live and st["state"] not in ("stopped", "done"):
                    return
                if st["state"] == "wait" and not st["chunks"]:
                    return                    # текст ещё готовится - пишет прогресс
                txt = S.status_text(st)
                if not live.gate and st["state"] == "wait" and live.speed.rtf > 1.0:
                    txt += " · возможны паузы"
                show(txt)
            self.ui(apply)
        live.on_status = status
        self._live, self._live_kind = live, kind
        self._live_buttons()
        return live

    def _speak_job(self, kind: str, text: str, engine: str, kw: dict, voice: str, *,
                   prepared: bool = False, fx_chain: str | None = None, note=None,
                   on_done=None, gate: bool = True, tempo: float | None = None,
                   on_spans=None):
        """Прочитать текст потоком: «Прочитать вслух», «Прослушать», «Послушать»."""
        show = (lambda t: note.configure(text=t)) if note is not None else (lambda t: None)
        show("Готовлю голос…")
        live = self._live_new(kind, engine, kw, voice, False, show, fx_chain, gate=gate)

        def work():
            audio, sr, err = None, 0, None
            try:
                audio, sr = S.speak(text, engine=engine, voice=voice, live=live,
                                    eng_kwargs=kw, prepared=prepared, tempo=tempo,
                                    on_spans=on_spans)
                if live.sink is None and not live.stopped:
                    self._play_plain(audio, sr, live.fx_chain or "")
                live.wait()
            except Cancelled:
                pass
            except Exception as e:
                err = e
                log_problem("прочитать вслух не удалось", e, show=False)

            def end():
                if on_done is not None and err is None and not live.stopped:
                    on_done(audio, sr)
                if err is not None:
                    show(f"Не вышло: {err}")
                elif live.first_sound is not None and not live.stopped:
                    show(f"Готово · первый звук через {live.first_sound:.1f} с"
                         .replace(".", ","))
                elif live.stopped:
                    show("Остановлено")
                if self._live is live:
                    self._live, self._live_kind = None, ""
                self._live_buttons()
            self.ui(end)
        threading.Thread(target=work, daemon=True).start()

    def _play_plain(self, audio, sr: int, chain: str):
        """Нет звуковой карты для потока - сыграть готовое файлом, как раньше."""
        import tempfile
        from core.assemble import write_wav
        d = tempfile.mkdtemp(prefix="herald_say_")
        raw, out = os.path.join(d, "raw.wav"), os.path.join(d, "play.mp3")
        write_wav(audio, sr, raw)
        polish_wav_to_mp3(raw, out, fx=chain)
        p = Player()
        if p.load(out):
            p.play()
            while p.playing and self._live is not None:
                time.sleep(0.1)
            p.stop()
        else:
            _open_path(out)

    def _clone_for(self, lang: str) -> str | None:
        v = self._voice_obj()
        if v is not None and not v.ready and v.language == lang:
            return v.id
        for x in self._voices:
            if not x.ready and x.language == lang:
                return x.id
        return v.id if v is not None and not v.ready else None

    def _say_warm(self):
        """Быстрый голос и ударения - заранее, пока вставляют текст: первый звук
        тогда приходит за доли секунды."""
        if getattr(self, "_warmed", False) or self.running:
            return
        self._warmed = True

        def work():
            try:
                from core import pipeline as P
                if S.ENGINE_LOCK.acquire(timeout=0):
                    try:
                        eng = P._loaded_engine("silero", {})
                        P._prepare_text("Проверка.", eng.stress_format, True, True, "ru")
                    finally:
                        S.ENGINE_LOCK.release()
            except Exception as e:
                log_problem("быстрый голос не загрузился заранее", e, show=False)
        threading.Thread(target=work, daemon=True).start()

    # ================= «Как будет прочитано» (3.7) =================
    def show_read_text(self):
        """Текст книги так, как он уйдёт в модель: числа словами, ударения.
        Правый щелчок по слову - «Исправить ударение…» (словарь пользователя)."""
        if not self.book_path:
            messagebox.showwarning("Нет книги", "Перетащите книгу в окно.")
            return
        path = self.book_path
        self.readtext_btn.configure(state="disabled", text="Готовлю текст…")

        def work():
            try:
                from core import pipeline as P
                saved = P.read_text_path(path)
                if os.path.exists(saved) and os.path.getmtime(saved) >= os.path.getmtime(path):
                    text = open(saved, encoding="utf-8").read()
                else:
                    text = P.prepare_read_text(path, ConvertOptions())
                self.ui(lambda: self._read_text_window(text))
            except Exception as e:
                log_problem("«как будет прочитано» не получилось", e, show=False)
                self.ui(lambda: messagebox.showerror("Ошибка", str(e)))
            finally:
                self.ui(lambda: self.readtext_btn.configure(state="normal",
                                                            text="Как будет прочитано"))
        threading.Thread(target=work, daemon=True).start()

    def _read_text_window(self, text: str):
        from core.stress import to_combining
        win = ctk.CTkToplevel(self)
        win.title("Как будет прочитано")
        win.geometry("760x680")
        T.label(win, "Ударение - знак над гласной. Правый щелчок по слову - «Исправить "
                     "ударение…»: исправление действует в следующих озвучках.",
                12, color=T.MUTED, anchor="w", justify="left", wraplength=700).pack(
            fill="x", padx=16, pady=(12, 6))
        box = ctk.CTkTextbox(win, font=T.font(15), wrap="word", border_width=1)
        box.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        box.insert("1.0", to_combining(text))
        box.configure(state="disabled")
        note = T.label(win, "", 12, color=T.OK, anchor="w")
        note.pack(fill="x", padx=16, pady=(0, 10))
        inner = getattr(box, "_textbox", box)
        import tkinter as tk

        def word_at(idx):
            line = inner.get(f"{idx} linestart", f"{idx} lineend")
            col = int(inner.index(idx).split(".")[1])
            ok = lambda c: c.isalpha() or c in "\u0301-"  # noqa: E731
            a = col
            while a > 0 and ok(line[a - 1]):
                a -= 1
            b = col
            while b < len(line) and ok(line[b]):
                b += 1
            return line[a:b].strip("-")

        def fix(word):
            plus = re.sub("(.)\u0301", "+\\1", word)
            dlg = ctk.CTkToplevel(win)
            dlg.title("Исправить ударение")
            T.label(dlg, "Поставьте «+» перед ударной гласной:", 13, anchor="w").pack(
                fill="x", padx=16, pady=(14, 6))
            ent = ctk.CTkEntry(dlg, width=260, font=T.font(15))
            ent.insert(0, plus.lower())
            ent.pack(padx=16)
            err = T.label(dlg, "", 12, color=T.WARN, anchor="w")
            err.pack(fill="x", padx=16, pady=(4, 0))

            def ok_(*_):
                from core.stress import set_user_stress
                try:
                    w = set_user_stress(ent.get())
                except ValueError as e:
                    err.configure(text=str(e))
                    return
                key = w.replace("+", "")
                box.configure(state="normal")
                body = box.get("1.0", "end-1c")
                pat = re.compile(r"(?<![\w\u0301])" + r"\u0301?".join(map(re.escape, key))
                                 + r"\u0301?(?![\w\u0301])", re.IGNORECASE)
                new = pat.sub(lambda m: _restress(m.group(), w), body)
                box.delete("1.0", "end")
                box.insert("1.0", new)
                box.configure(state="disabled")
                note.configure(text=f"Сохранено: {to_combining(w)} - действует в следующих озвучках")
                dlg.destroy()
            T.button(dlg, "Сохранить", "save", kind="primary", height=32,
                     command=ok_).pack(pady=(8, 14))
            dlg._entry, dlg._ok = ent, ok_
            win._dlg = dlg
            ent.bind("<Return>", ok_)
            dlg.bind("<Escape>", lambda e: dlg.destroy())
            ent.focus_set()

        menu = tk.Menu(inner, tearoff=0)

        def popup(event):
            idx = inner.index(f"@{event.x},{event.y}")
            w = word_at(idx)
            if not w:
                return "break"
            menu.delete(0, "end")
            menu.add_command(label=f"Исправить ударение: «{w}»…", command=lambda: fix(w))
            menu.tk_popup(event.x_root, event.y_root)
            return "break"
        for seq in ("<Button-2>", "<Button-3>"):
            inner.bind(seq, popup)
        if sys.platform == "darwin":
            inner.bind("<Control-Button-1>", popup)
        win.bind("<Escape>", lambda e: win.destroy())
        win._word_at, win._fix, win._box, win._note = word_at, fix, box, note   # для проверок
        self._read_win = win

    # ================= вкладка «Прочитать вслух» (3.5) =================
    SAY_KEYS = {"a": "all", "c": "copy", "v": "paste", "x": "cut", "z": "undo"}
    # та же клавиша при русской раскладке: Tk на Mac отдаёт кириллическое имя
    SAY_CYR = {"Cyrillic_ef": "a", "Cyrillic_es": "c", "Cyrillic_em": "v",
               "Cyrillic_che": "x", "Cyrillic_ya": "z"}
    # и код клавиши - на случай, если имени нет: Mac (виртуальный код в старших
    # байтах), Windows (VK), Linux X11
    SAY_CODES = {"darwin": {0: "a", 8: "c", 9: "v", 7: "x", 6: "z"},
                 "win32": {65: "a", 67: "c", 86: "v", 88: "x", 90: "z"},
                 "linux": {38: "a", 54: "c", 55: "v", 53: "x", 52: "z"}}

    def _build_say_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=28, pady=(22, 14))
        T.label(head, "Прочитать вслух", 22, bold=True, anchor="w").pack(side="left")
        zoom = ctk.CTkFrame(head, fg_color="transparent")
        zoom.pack(side="right")
        T.button(zoom, "A−", None, kind="ghost", height=28, width=44, size=13,
                 command=lambda: self._say_zoom_by(-10)).pack(side="left")
        self.say_zoom_lbl = T.label(zoom, "", 12, color=T.MUTED, width=48)
        self.say_zoom_lbl.pack(side="left")
        T.button(zoom, "A+", None, kind="ghost", height=28, width=44, size=13,
                 command=lambda: self._say_zoom_by(10)).pack(side="left")

        left = T.card(page)
        left.grid(row=1, column=0, sticky="nsew", padx=(28, 8), pady=(0, 20))
        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(12, 6))
        T.button(bar, "Из буфера", "copy", kind="ghost", height=30, size=13, width=120,
                 command=self._say_paste).pack(side="left")
        T.button(bar, "Открыть файл…", "folder-open", kind="ghost", height=30, size=13,
                 width=150, command=self._say_open).pack(side="left", padx=6)
        T.button(bar, "Очистить", "x", kind="ghost", height=30, size=13, width=110,
                 command=self._say_clear).pack(side="left")
        self.say_count = T.label(bar, "", 12, color=T.MUTED, anchor="e")
        self.say_count.pack(side="right")
        self.say_box = ctk.CTkTextbox(left, font=T.font(14), border_width=1, wrap="word",
                                      undo=True)
        self.say_box.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        self._say_bind_keys()
        inner = getattr(self.say_box, "_textbox", self.say_box)
        # светлая заливка и тёмный текст в обеих темах (3.6: на тёмной теме
        # прежняя заливка закрывала текст); контраст 14,4:1 - проверено в test_gui
        inner.tag_configure("say_cur", background=SAY_HL_BG, foreground=SAY_HL_FG)
        inner.bind("<<Modified>>", lambda e: self._say_text_changed())
        inner.bind("<ButtonRelease-1>", self._say_click)
        inner.bind("<FocusIn>", lambda e: self._say_warm())
        ctrl = ctk.CTkFrame(left, fg_color="transparent")
        ctrl.pack(fill="x", padx=14, pady=(0, 12))
        self.say_btn = T.button(ctrl, "Прочитать", "play", kind="primary", height=34, size=13,
                                width=140, command=self.do_say)
        self.say_btn.pack(side="left")
        self.say_pause_btn = T.button(ctrl, "", "pause", height=34, width=44,
                                      command=self._say_pause)
        self.say_pause_btn.pack(side="left", padx=(6, 0))
        self.say_stop_btn = T.button(ctrl, "", "square", height=34, width=44,
                                     command=self._say_stop)
        self.say_stop_btn.pack(side="left", padx=(6, 0))
        self.say_save_btn = T.button(ctrl, "Сохранить", "save", kind="ghost", height=34,
                                     size=13, width=120, command=self._say_save)
        self.say_save_btn.configure(state="disabled")
        self.say_save_btn.pack(side="right")
        # строка состояния - отдельной строкой: между кнопками она обрезалась (3.6)
        self.say_note = T.label(left, "", 12, color=T.MUTED, anchor="w")
        self.say_note.pack(fill="x", padx=16, pady=(0, 10))

        right = T.card(page)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 28), pady=(0, 20))
        r = ctk.CTkFrame(right, fg_color="transparent")
        r.pack(fill="both", expand=True, padx=18, pady=16)
        ctk.CTkFrame(r, width=290, height=0, fg_color="transparent").pack()
        T.section(r, "Голос").pack(fill="x")
        # «Быстрый / Клон» - только если клоны показаны (Настройки, 3.6)
        self.say_mode_box = ctk.CTkFrame(r, fg_color="transparent")
        self.say_mode_box.pack(fill="x")
        self.say_mode = ctk.CTkSegmentedButton(
            self.say_mode_box, values=["Быстрый", "Клон"], height=30, font=T.font(13),
            selected_color=T.ACCENT, selected_hover_color=T.ACCENT_HOVER,
            command=lambda v: self._say_on_kind(v))
        self.say_mode.pack(fill="x", pady=(6, 4))
        self.say_voice_var = ctk.StringVar(value="")
        self.say_voice_menu = ctk.CTkOptionMenu(r, variable=self.say_voice_var, values=[""],
                                                height=30, font=T.font(13),
                                                dropdown_font=T.font(13),
                                                command=lambda *_: self._say_on_voice())
        self.say_voice_menu.pack(fill="x")
        # «Сразу / Без пауз» - только для клона: быстрые голоса и так без пауз
        self.say_start_box = ctk.CTkFrame(r, fg_color="transparent")
        self.say_start_box.pack(fill="x")
        T.section(self.say_start_box, "Начало").pack(fill="x", pady=(14, 0))
        sv = self.cfg.get("say_start")
        self.say_start_var = ctk.StringVar(value=sv if sv in ("Сразу", "Без пауз") else "Сразу")
        for val, text in (("Сразу", "Начать сразу (между фразами возможны паузы)"),
                          ("Без пауз", "Подождать и читать без остановок")):
            ctk.CTkRadioButton(self.say_start_box, text=text, value=val,
                               variable=self.say_start_var, font=T.font(13),
                               fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                               border_color=T.BORDER_STRONG, text_color=T.TEXT,
                               command=self._say_on_start).pack(anchor="w", pady=(6, 0))
        self.say_sound_head = T.section(r, "Звучание")
        self.say_sound_head.pack(fill="x", pady=(14, 0))
        self._say_fx_vars = {}
        for title, key, lo, hi, unit in (("Темп", "speed", SAY_TEMPO[0], SAY_TEMPO[1], "×"),
                                         ("Высота", "pitch", -4, 4, " полутона"),
                                         ("Басы", "bass", -6, 6, " дБ"),
                                         ("Яркость", "brightness", -6, 6, " дБ")):
            line = ctk.CTkFrame(r, fg_color="transparent")
            line.pack(fill="x", pady=3)
            T.label(line, title, 13, anchor="w", width=70).pack(side="left")
            val = T.label(line, "", 12, color=T.MUTED, width=80, anchor="e")
            val.pack(side="right")
            steps = 20 if key == "speed" else int((hi - lo) * 4)
            sl = ctk.CTkSlider(line, from_=lo, to=hi, number_of_steps=steps,
                               button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
                               progress_color=T.ACCENT, fg_color=T.SURFACE2,
                               command=lambda _v, k=key: self._say_fx_changed(k))
            sl.pack(side="left", fill="x", expand=True, padx=6)
            self._say_fx_vars[key] = (sl, val, unit)
        bb = ctk.CTkFrame(r, fg_color="transparent")
        bb.pack(fill="x", pady=(10, 0))
        T.button(bb, "Сбросить", "refresh-cw", kind="ghost", height=30, size=13, width=110,
                 command=self._say_fx_reset).pack(side="left")
        self.say_fx_save_btn = T.button(bb, "Сохранить для голоса", "save", height=30, size=13,
                                        width=170, command=self._say_fx_save)
        self.say_fx_save_btn.pack(side="right")
        self.say_forecast = T.label(r, "", 12, color=T.MUTED, anchor="w", justify="left",
                                    wraplength=290)
        self.say_forecast.pack(fill="x", pady=(14, 0))
        self._say_zoom = int(self.cfg.get("say_zoom", 100))
        self._say_zoom_by(0)
        self._say_spans: list = []
        self._say_offset = 0
        self._say_from = 0
        self.say_mode.set(self.cfg.get("say_mode", "Быстрый")
                          if self.cfg.get("say_mode") in ("Быстрый", "Клон") else "Быстрый")
        self._say_buttons()
        return page

    # --- клавиши: Cmd/Ctrl + A C V X Z при любой раскладке ---
    def _say_bind_keys(self):
        inner = getattr(self.say_box, "_textbox", self.say_box)
        mods = ("Command", "Control") if sys.platform == "darwin" else ("Control",)
        for m in mods:
            inner.bind(f"<{m}-KeyPress>", self._say_key, add=False)
        import tkinter as tk
        menu = tk.Menu(inner, tearoff=0)
        for label, act in (("Вырезать", "cut"), ("Копировать", "copy"), ("Вставить", "paste"),
                           (None, None), ("Выделить всё", "all")):
            if label is None:
                menu.add_separator()
            else:
                menu.add_command(label=label, command=lambda a=act: self._say_edit(a))

        def popup(event):
            menu.tk_popup(event.x_root, event.y_root)
            return "break"
        for seq in ("<Button-2>", "<Button-3>"):
            inner.bind(seq, popup)
        if sys.platform == "darwin":
            inner.bind("<Control-Button-1>", popup)

    def _say_key_action(self, event) -> str | None:
        ks = event.keysym or ""
        if ks.startswith("Cyrillic_"):
            ks = "Cyrillic_" + ks[9:].lower()
        k = ks.lower() if len(ks) == 1 else self.SAY_CYR.get(ks)
        if k not in self.SAY_KEYS:
            plat = "darwin" if sys.platform == "darwin" else (
                "win32" if sys.platform.startswith("win") else "linux")
            code = int(event.keycode or 0)
            k = self.SAY_CODES[plat].get(code >> 24 if plat == "darwin" and code > 0xFFFFFF
                                         else code)
        return self.SAY_KEYS.get(k) if k else None

    def _say_key(self, event):
        act = self._say_key_action(event)
        if act is None:
            return None
        if act == "undo" and event.state & 0x1:      # с Shift - вернуть
            act = "redo"
        self._say_edit(act)
        return "break"

    def _say_edit(self, act: str):
        inner = getattr(self.say_box, "_textbox", self.say_box)
        inner.focus_set()
        ev = {"copy": "<<Copy>>", "paste": "<<Paste>>", "cut": "<<Cut>>",
              "undo": "<<Undo>>", "redo": "<<Redo>>"}.get(act)
        if act == "all":
            inner.tag_add("sel", "1.0", "end-1c")
            inner.mark_set("insert", "end-1c")
        elif ev:
            try:
                inner.event_generate(ev)
            except Exception:
                pass                                 # отменять нечего
        self._say_text_changed()

    # --- текст ---
    def _say_text(self) -> str:
        return self.say_box.get("1.0", "end-1c")

    def _say_text_changed(self):
        inner = getattr(self.say_box, "_textbox", self.say_box)
        try:
            inner.edit_modified(False)
        except Exception:
            pass
        n = len(self._say_text().strip())
        self.say_count.configure(text=f"{n:,} знаков".replace(",", " ") if n else "")
        self._say_forecast()

    def _say_paste(self):
        try:
            t = self.clipboard_get()
        except Exception:
            t = ""
        if t.strip():
            self.say_box.delete("1.0", "end")
            self.say_box.insert("1.0", t.strip())
            self._say_text_changed()
            self._say_warm()
        else:
            self.say_note.configure(text="В буфере нет текста")

    def _say_open(self):
        path = filedialog.askopenfilename(
            filetypes=[("Тексты и книги", "*.txt *.md *.docx *.fb2 *.epub *.pdf *.zip"),
                       ("Все файлы", "*.*")])
        if not path:
            return
        try:
            from core.readers import read_document
            doc = read_document(path)
            text = doc.text or "\n\n".join(
                (f"{c.title}\n\n" if getattr(c, "title", "") else "") + c.text
                for c in (doc.chapters or []))
        except Exception as e:
            messagebox.showerror("Не открылось", str(e))
            return
        self.say_box.delete("1.0", "end")
        self.say_box.insert("1.0", text.strip())
        self._say_text_changed()

    def _say_clear(self):
        self._say_stop()
        self.say_box.delete("1.0", "end")
        self._say_text_changed()

    def _say_zoom_by(self, d: int):
        self._say_zoom = max(70, min(200, self._say_zoom + d))
        self.say_zoom_lbl.configure(text=f"{self._say_zoom}%")
        self.say_box.configure(font=T.font(round(14 * self._say_zoom / 100)))
        if d:
            self.cfg["say_zoom"] = self._say_zoom
            save_config(self.cfg)

    # --- голос ---
    def _say_clones(self) -> bool:
        return bool(self.cfg.get("say_clones", False))

    def _say_voices(self, kind: str) -> list[_V]:
        if kind != "Быстрый" and not self._say_clones():
            kind = "Быстрый"
        return [v for v in self._voices if v.ready == (kind == "Быстрый")]

    @staticmethod
    def _say_label(v: _V) -> str:
        """«Евгений · муж. · русский» - имя, пол, язык (3.6)."""
        name = v.title.split(" (")[0]
        g = GENDER.get(v.gender, "") if v.ready else "клон"
        return " · ".join(x for x in (name, g, LANG_MARK.get(v.language, v.language)) if x)

    def _say_on_start(self):
        self.cfg["say_start"] = self.say_start_var.get()
        save_config(self.cfg)
        self._say_forecast()

    def _say_layout(self):
        """Показать «Быстрый / Клон» и «Начало» только там, где они нужны."""
        if self._say_clones():
            self.say_mode_box.pack(fill="x", before=self.say_voice_menu)
        else:
            self.say_mode_box.pack_forget()
            if self.say_mode.get() != "Быстрый":
                self.say_mode.set("Быстрый")
        v = self._say_voice()
        if v is not None and not v.ready:
            self.say_start_box.pack(fill="x", before=self.say_sound_head)
        else:
            self.say_start_box.pack_forget()

    def _say_on_kind(self, kind: str, remember: bool = True):
        if remember:
            self.cfg["say_mode"] = kind
            save_config(self.cfg)
        vs = self._say_voices(kind)
        labels = [self._say_label(v) for v in vs] or ["(нет голосов)"]
        self.say_voice_menu.configure(values=labels)
        want = self.cfg.get("say_fast_voice" if kind == "Быстрый" else "say_clone_voice")
        if kind == "Быстрый" and not want:
            want = FAST_VOICE
        if kind == "Клон" and not want:
            v = self._voice_obj()
            want = v.id if v is not None and not v.ready else None
        pick = next((v for v in vs if v.id == want), vs[0] if vs else None)
        self.say_voice_var.set(self._say_label(pick) if pick else labels[0])
        self._say_on_voice(remember=False)

    def _say_voice(self) -> _V | None:
        lab = self.say_voice_var.get()
        return next((v for v in self._voices if self._say_label(v) == lab), None)

    def _say_on_voice(self, remember: bool = True):
        v = self._say_voice()
        if v is not None and remember:
            self.cfg["say_fast_voice" if v.ready else "say_clone_voice"] = v.id
            save_config(self.cfg)
        fx = load_fx(default_voices_dir(), v.id) if v is not None else VoiceFX()
        for k, (sl, _l, _u) in self._say_fx_vars.items():
            sl.set(getattr(fx, k))          # сохранённое для голоса - начальное положение
            self._say_fx_label(k)
        self._say_layout()
        self._say_forecast()

    def _say_fx(self) -> VoiceFX:
        return VoiceFX(**{k: round(sl.get(), 2 if k == "speed" else 1)
                          for k, (sl, _l, _u) in self._say_fx_vars.items()})

    def _say_fx_label(self, key: str):
        sl, lbl, unit = self._say_fx_vars[key]
        if key == "speed":
            v = round(sl.get(), 2)
            lbl.configure(text="как есть" if abs(v - 1) < 0.01 else f"{v:.2f}{unit}")
        else:
            v = round(sl.get(), 1)
            lbl.configure(text="нет" if v == 0 else f"{v:+.1f}{unit}")

    def _say_fx_changed(self, key: str):
        """Ползунок: прогноз - сразу; во время чтения высота, басы, яркость -
        с ближайшего куска, темп - с ещё не посчитанного."""
        self._say_fx_label(key)
        live = self._live if self._live_kind == "say" else None
        if live is not None:
            fx, v = self._say_fx(), self._say_voice()
            if v is not None and v.ready:
                live.fx_chain = S.with_tempo(fx.filter_chain(), fx.speed)
            else:
                live.fx_chain = fx.filter_chain()
                live.tempo = fx.speed
        self._say_forecast()

    def _say_fx_reset(self):
        for k, (sl, _l, _u) in self._say_fx_vars.items():
            sl.set(1.0 if k == "speed" else 0)
            self._say_fx_changed(k)

    def _say_fx_save(self):
        v = self._say_voice()
        if v is None:
            return
        save_fx(default_voices_dir(), v.id, self._say_fx())
        self.say_note.configure(text=f"Звучание сохранено для «{v.title}»")
        if self._fx_voice == v.id:
            self.select_fx_voice(v.id)       # «Голоса» показывают то же
        self._refresh_voice_rows()

    # --- прогноз: сколько звука и когда начнётся ---
    def _say_plan(self):
        """(движок, параметры, голос, язык, подсказка) для текущего выбора."""
        from core.normalize import detect_language
        text = self._say_text()
        lang = detect_language(text) if text.strip() else "ru"
        v = self._say_voice()
        hint = ""
        if v is not None and v.ready and v.language == lang:
            return v.engine, {}, v.id, lang, hint
        if v is not None and v.ready:
            other = next((x for x in self._voices if x.ready and x.language == lang), None)
            if other is not None:              # другой язык - быстрый голос этого языка
                return other.engine, {}, other.id, lang, f"Текст {LANG_MARK.get(lang, lang)} - читает {other.title}. "
            hint = f"Быстрого голоса на языке текста нет - читает клон. "
            vid = self._clone_for(lang)
        else:
            vid = v.id if v is not None else self._clone_for(lang)
        kw = self._prof_kwargs()
        if lang != "ru":
            kw.pop("variant", None)
        return clone_engine(), kw, vid, lang, hint

    def _say_speed(self, engine, kw, vid):
        prior = (profiles.SILERO.rtf if not _clone_engine(engine)
                 else profiles.rtf_for(self.profile(), self._accel))
        return S.make_speed(engine, dict(kw, language=self._say_plan()[3])
                            if _clone_engine(engine) else kw, vid, False, prior, self._accel)

    def _say_forecast(self):
        if not hasattr(self, "say_forecast"):
            return
        text = self._say_text()[self._say_from:].strip()
        if not text:
            self.say_forecast.configure(text="")
            return
        engine, kw, vid, _lang, hint = self._say_plan()
        if not vid:
            self.say_forecast.configure(text=hint + "Нет голоса-клона - добавьте образец.")
            return
        sp = self._say_speed(engine, kw, vid)
        tempo = self._say_fx().speed
        clone = _clone_engine(engine)
        from core.stream import _letters
        size = 187 if clone else 800
        parts, cur = [], 0
        for w in text.split():
            cur += len(w) + 1
            if cur >= size:
                parts.append(cur)
                cur = 0
        if cur:
            parts.append(cur)
        first = min(parts[0], S.FIRST_MAX) if parts else 0
        letters = _letters(text)
        audio = sp.audio_sec(letters, tempo if clone else 1.0) / (1.0 if clone else tempo)
        rest = []
        for n in ([first, parts[0] - first] if parts and parts[0] > first else [first]) + parts[1:]:
            d = sp.audio_sec(int(n * 0.8), tempo if clone else 1.0)
            rest.append((sp.synth_sec(d), d))
        if self.say_start_var.get() == "Сразу":
            wait = rest[0][0] if rest else 0.0
            maybe = " · возможны паузы" if clone and sp.rtf > 1.0 else ""
        else:
            wait = S.wait_needed(0.0, rest)
            maybe = ""
        self.say_forecast.configure(text=(
            f"{hint}Звука ~{_fmt_hms(audio)} · звук через ~{max(1, round(wait))} с{maybe}"))

    # --- чтение ---
    def _say_buttons(self):
        on = self._live_kind == "say" and self._live is not None
        paused = on and self._live.paused
        self.say_btn.configure(text="Читаю…" if on and not paused else "Прочитать",
                               state="disabled" if self.running or (on and not paused) else "normal")
        T.restyle(self.say_btn, "primary", "play")
        self.say_pause_btn.configure(state="normal" if on and not paused else "disabled")
        self.say_stop_btn.configure(state="normal" if on else "disabled")

    def do_say(self):
        """▶: прочитать с начала (или с выбранного предложения); на паузе - продолжить."""
        if self._live_kind == "say" and self._live is not None:
            if self._live.paused:
                self._live.resume()
                self._say_buttons()
            return
        if self.running:
            return
        full = self._say_text()
        start = self._say_from
        text = full[start:]
        if not text.strip():
            self.say_note.configure(text="Вставьте текст или нажмите «Из буфера»")
            return
        engine, kw, vid, _lang, hint = self._say_plan()
        if not vid:
            self.say_note.configure(text="Нет голоса-клона - добавьте образец в «Голосах»")
            return
        fx = self._say_fx()
        gate = self.say_start_var.get() == "Без пауз"
        self.say_save_btn.configure(state="disabled")
        chain = fx.filter_chain()

        def done(audio, sr):
            self._say_last = (audio, sr, S.with_tempo(chain, fx.speed) if not _clone_engine(engine)
                              else chain)
            self.say_save_btn.configure(state="normal")
        if hint:
            self.say_note.configure(text=hint)

        def spans(sp):
            self._say_spans = [(a + start, b + start) for a, b in sp]
        self._speak_job("say", text, engine, kw, vid, note=self.say_note, on_done=done,
                        gate=gate, tempo=fx.speed, fx_chain=chain, on_spans=spans)
        self._say_buttons()
        self._say_tick()

    def _say_pause(self):
        if self._live_kind == "say" and self._live is not None and not self._live.paused:
            self._live.pause()
            self.say_note.configure(text="Пауза · ▶ - продолжить")
        self._say_buttons()

    def _say_stop(self):
        if self._live_kind == "say":
            self._live_stop()
        self._say_highlight(None)
        self._say_buttons()

    def _say_tick(self):
        """Подсветка звучащего предложения (по готовым кускам)."""
        live = self._live if self._live_kind == "say" else None
        if live is None:
            self._say_highlight(None)
            self._say_buttons()
            return
        k = live.playing_chunk()
        self._say_highlight(self._say_spans[k] if k is not None and k < len(self._say_spans)
                            else None)
        self.after(200, self._say_tick)

    def _say_highlight(self, span):
        inner = getattr(self.say_box, "_textbox", self.say_box)
        inner.tag_remove("say_cur", "1.0", "end")
        if span:
            a, b = span
            inner.tag_add("say_cur", f"1.0+{a}c", f"1.0+{b}c")
            inner.see(f"1.0+{a}c")

    def _say_click(self, event):
        """Щелчок по предложению - читать с него (во время чтения - сразу)."""
        inner = getattr(self.say_box, "_textbox", self.say_box)
        try:
            if inner.tag_ranges("sel"):
                return                                   # выделяли текст - не мешаем
            pos = int(inner.count("1.0", inner.index(f"@{event.x},{event.y}"), "chars")[0] or 0)
        except Exception:
            return
        spans = S.sentence_spans(self._say_text())
        hit = next(((a, b) for a, b in spans if a <= pos < b), None)
        if hit is None:
            return
        self._say_from = hit[0]
        self._say_highlight(hit)
        self._say_forecast()
        if self._live_kind == "say" and self._live is not None:
            self._live_stop()
            self.after(50, self.do_say)
        else:
            self.say_note.configure(text="Начну с выделенного предложения · ▶")

    def _say_save(self):
        if not self._say_last:
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp3",
                                            filetypes=[("mp3", "*.mp3")],
                                            initialfile="Прочитано вслух.mp3")
        if not path:
            return
        audio, sr, chain = self._say_last
        import tempfile
        from core.assemble import write_wav
        raw = os.path.join(tempfile.mkdtemp(prefix="herald_say_"), "raw.wav")
        write_wav(audio, sr, raw)
        try:
            polish_wav_to_mp3(raw, path, fx=chain)
            self.say_note.configure(text=f"Сохранено: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _profile_listen(self, key: str):
        """Настройки, «Послушать»: фраза выбранным голосом и этим движком."""
        kind = "prof:" + key
        if self._live_kind == kind:
            self._live_stop()
            return
        v = self._voice_obj()
        if v is None or v.ready:
            self.prof_note.configure(text="Движок слышно на клоне: выберите голос-клон на главной")
            return
        p = profiles.get(key)
        kw = self._prof_kwargs(p)
        if v.language != "ru":
            if "variant" in kw:
                self.prof_note.configure(text="У ESpeech только русский голос")
                return
        text = self.FX_PHRASE.get(v.language, self.FX_PHRASE["ru"])
        self._speak_job(kind, text, clone_engine(), kw, v.id, prepared=True,
                        note=self.prof_note)

    # --- озвучка для других программ ---
    def _srv_sync(self):
        """Голос и движок для сервиса - из окна (читаются в главном потоке)."""
        v = self._voice_obj()
        self._srv_voice = v.id if v is not None and not v.ready else self._clone_for("ru")
        self._srv_kwargs = self._prof_kwargs()

    def _srv_toggle(self):
        self.cfg["speak_server"] = bool(self.srv_var.get())
        save_config(self.cfg)
        self._srv_apply()

    def _srv_apply(self):
        from core.speak_server import SpeakServer
        try:
            port = int(self.srv_port.get().strip())
            if not 1024 <= port <= 65535:
                raise ValueError
        except ValueError:
            self.srv_note.configure(text="Порт - число от 1024 до 65535", text_color=T.WARN)
            return
        idle = int(self.srv_idle.get().split()[0])
        self.cfg.update(speak_port=port, speak_idle=idle)
        save_config(self.cfg)
        on = bool(self.srv_var.get())
        if self._srv is not None and (not on or self._srv.port != port):
            self._srv.stop()
            self._srv = None
        if not on:
            self.srv_note.configure(text="Выключен", text_color=T.MUTED)
            return
        if self._srv is None:
            srv = SpeakServer(port=port, idle_min=idle,
                              default_voice=lambda: self._srv_voice,
                              profile_kwargs=lambda: dict(self._srv_kwargs),
                              log=lambda m: self._logln(m))
            try:
                srv.start()
            except OSError as e:
                self.srv_note.configure(text=f"Порт {port} занят ({e.strerror})",
                                        text_color=T.WARN)
                return
            self._srv = srv
        self._srv.idle_min = idle
        self.srv_note.configure(text=f"Работает: {self._srv.url}/speak", text_color=T.OK)

    def _fx_save(self):
        if not self._fx_voice:
            return
        save_fx(default_voices_dir(), self._fx_voice, self._fx_current())
        self.fx_note.configure(text=f"Сохранено для «{self._fx_voice}»")
        self._refresh_voice_rows()
        self.show_estimate()           # темп влияет на длину и прогноз

    def load_sample(self):
        p = filedialog.askopenfilename(
            title="Запись нужного диктора (лучше аудиокнига, связная проза)",
            filetypes=[("Аудио", "*.mp3 *.wav *.m4a *.m4b *.flac *.ogg"), ("Все файлы", "*.*")])
        if not p:
            return
        name = ctk.CTkInputDialog(title="Имя голоса", text="Как назвать этот голос?").get_input()
        if not name or not name.strip():
            return
        self._set_busy(True, "Слушаю запись и выбираю отрывок для образца…")

        def work():
            try:
                vid = get_engine(clone_engine()).prepare_reference(p, name=name.strip())
                self.ui(lambda: (self._set_busy(False, f"Голос готов: {vid}"),
                                 self.cfg.update(voice=vid), self.refresh_voices(),
                                 self._refresh_voice_rows(), self.select_fx_voice(vid)))
            except Exception as e:
                self._fail(e, "Не удалось подготовить голос")
        threading.Thread(target=work, daemon=True).start()

    # ================= книга =================
    def on_drop(self, event):
        paths = [p for p in self.tk.splitlist(event.data) if p.lower().endswith(BOOK_EXT)]
        if not paths:
            messagebox.showwarning("Не та книга", "Поддерживаются: " + ", ".join(BOOK_EXT))
            return
        if not self.running:
            self._set_book(paths[0])

    def choose_book(self):
        if self.running:
            return
        p = filedialog.askopenfilename(
            title="Книга", filetypes=[("Книги", " ".join("*" + e for e in BOOK_EXT)),
                                      ("Все файлы", "*.*")])
        if p:
            self._set_book(p)

    def clear_book(self):
        if self.running:
            return
        self.book_path = None
        self.est = None
        self.info.pack_forget()
        self.empty.pack(fill="both", expand=True, padx=14, pady=14)
        self.progress.set(0)
        self.status.configure(text="Перетащите книгу - и нажмите «Озвучить книгу».")
        self.timer.configure(text=_fmt_clock(0))
        self.log.delete("1.0", "end")
        self._show_resume(None, None)

    def _set_book(self, p: str):
        self.book_path = p
        self.empty.pack_forget()
        self.info.pack(fill="both", expand=True)
        self.book_title.configure(text=os.path.splitext(os.path.basename(p))[0])
        self.book_author.configure(text=os.path.dirname(p))
        self._clear_parts()
        self.reestimate()

    def single_file(self) -> bool:
        return self.len_var.get() == SINGLE_FILE

    def file_minutes(self) -> float:
        if self.single_file():
            return SINGLE_PART_MINUTES
        return float(self.len_var.get().split()[0])

    def reestimate(self):
        if not self.book_path:
            return
        self.est = None
        self.est_label.configure(text="Считаю объём книги…")
        path, minutes = self.book_path, self.file_minutes()

        def work():
            try:
                est = estimate_book(path, minutes)
                self.ui(lambda: self._got_estimate(path, est))
            except Exception as e:
                log_problem("книга не прочиталась", e, show=False)
                self.ui(lambda: self.est_label.configure(text=f"Не удалось прочитать книгу: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def _got_estimate(self, path, est):
        if path != self.book_path:
            return
        self.est = est
        from core.chapters import clean_book_title
        self.book_title.configure(text=clean_book_title(est.title) if est.title
                                  else os.path.basename(path))
        self.book_author.configure(text=est.author or "автор не указан")
        self.refresh_voices()
        self.show_estimate()
        self._load_existing_parts()

    def show_estimate(self):
        if not self.est:
            return
        e = self.est
        v = self._voice_obj()
        rate = 1.0
        if v and not v.ready:
            rate = load_fx(default_voices_dir(), v.id).speed or 1.0
        audio_sec = e.audio_sec / rate * float(self.cfg.get("audio_k", {}).get(v.id, 1.0)
                                               if v else 1.0)
        kbps = {"mp3 192 кбит/с (обычный)": 192, "mp3 320 кбит/с": 320}.get(self.fmt_var.get(), 750)
        mb = audio_sec * kbps / 8 / 1024
        size = f"{mb / 1024:.1f} ГБ" if mb >= 1024 else f"{mb:.0f} МБ"
        lang = {"ru": "русская книга", "en": "английская книга"}.get(getattr(e, "language", "ru"), "")
        parts = "один файл" if self.single_file() else _files(e.chapters)
        wait = audio_sec * self._rtf()
        night = "  Удобно поставить на ночь." if wait > 3 * 3600 else ""
        self.est_label.configure(text=(
            f"{lang} · {parts} · звука {_fmt_hms(audio_sec)}\n"
            f"Ждать около {_fmt_hms(wait)} · на диске ~{size}.{night}"))
        self._check_resume()

    # --- список частей ---
    def _clear_parts(self):
        self._close_player()
        for w in self.rows.winfo_children():
            if w is not self.no_parts:
                w.destroy()
        self.no_parts.pack(fill="x", padx=14, pady=10)
        self._n_parts = 0

    def _add_part(self, title: str, path: str | None, note: str = ""):
        self.no_parts.pack_forget()
        PartRow(self.rows, title, path, note, on_listen=self.listen).pack(fill="x")
        self._n_parts = getattr(self, "_n_parts", 0) + 1
        self.parts_title.configure(text=f"Готовые части · {self._n_parts}")

    def _load_existing_parts(self):
        """Части, оставшиеся с прошлого раза, - сразу в список."""
        self._clear_parts()
        try:
            d = output_dir_for(self.book_path)
        except Exception:
            return
        if not os.path.isdir(d):
            return
        exts = tuple(v[0] for v in FORMATS.values())
        for f in sorted(os.listdir(d)):
            if f.startswith("Часть ") and f.endswith(exts) or (
                    f.endswith(exts) and not f.startswith(".")):
                self._add_part(os.path.splitext(f)[0], os.path.join(d, f), "с прошлого раза")

    # --- мини-плеер ---
    def listen(self, row: PartRow):
        """«Слушать»: открыть плеер в строке части (другая строка закрывается)."""
        if self._active_row is not row:
            self._close_player()
            if not self.player.load(row.path):
                _open_path(row.path)          # плеер недоступен - как раньше, снаружи
                return
            self._active_row = row
            row.open_transport(self.player)
            self.player.play()
            self._tick_player()
        else:
            self.player.toggle()
        row.sync(self.player)

    def _close_player(self):
        self.player.stop()
        row, self._active_row = self._active_row, None
        if row is not None and row.winfo_exists():
            row.close_transport()

    def _tick_player(self):
        row = self._active_row
        if row is None:
            return
        if not row.winfo_exists():
            self._close_player()
            return
        row.sync(self.player)
        self.after(250, self._tick_player)

    # --- продолжение ---
    def _check_resume(self):
        if not self.book_path or self.running:
            return
        path, voice = self.book_path, self.current_voice()
        (engine, kw), opts = self._eng_choice(), self._conv_options()
        lang = getattr(self.est, "language", None)

        def work():
            info = resume_info(path, engine=engine, eng_kwargs=kw, voice=voice, options=opts,
                               language=lang) if voice else None
            self.ui(lambda: self._show_resume(path, info))
        threading.Thread(target=work, daemon=True).start()

    def _show_resume(self, path, info):
        if path != self.book_path or (self.running and info):
            return
        self._resume = info
        if self.running:
            return
        if info:
            self.start_btn.configure(text=f"Продолжить ({info['done']} из {info['total']})")
            if not self.restart_btn.winfo_manager():
                self.restart_btn.pack(side="right", padx=(0, 8), after=self.start_btn)
        else:
            self.start_btn.configure(text="Озвучить книгу")
            self.restart_btn.pack_forget()

    def do_restart(self):
        if self.running or not self.book_path:
            return
        try:
            forget_progress(self.book_path)
        except Exception as e:
            log_problem("не удалось стереть сохранённые части", e)
        self._show_resume(self.book_path, None)
        self._clear_parts()
        self.do_start()

    # ================= работа =================
    def _eng_choice(self) -> tuple[str, dict]:
        """(движок, параметры): готовый голос - Silero, клон - F5 по профилю.
        Английскому голосу - без ESpeech (у него веса только русские)."""
        v = self._voice_obj()
        if v is not None and v.ready:
            return v.engine, {}
        kw = self._prof_kwargs()
        if v is not None and v.language != "ru":
            kw.pop("variant", None)
        return clone_engine(), kw

    # совместимость с тестами и CLI
    def _eng_kwargs(self) -> dict:
        return self._eng_choice()[1]

    def _conv_options(self) -> ConvertOptions:
        pauses = {k: self._pause(k) for k in self._pause_vars}
        return ConvertOptions(make_m4b=False, make_chapter_mp3=True,
                              minutes_per_file=self.file_minutes(),
                              audio_format=self.fmt_var.get(), single_file=self.single_file(),
                              strict_words=bool(self.strict_var.get()),
                              sound_check=bool(self.sound_var.get()), **pauses)

    def _ready(self) -> bool:
        if not self.book_path or not os.path.isfile(self.book_path):
            messagebox.showwarning("Нет книги", "Перетащите книгу в окно.")
            return False
        if not self.current_voice():
            messagebox.showwarning("Нет голоса", "Добавьте голос на странице «Голоса».")
            return False
        return True

    def _set_busy(self, busy: bool, status: str = ""):
        self.running = busy
        st = "disabled" if busy else "normal"
        for w in self._setting_widgets + [self.preview_btn, self.clear_btn, self.add_voice_btn]:
            w.configure(state=st)
        for r in self._profile_radios + getattr(self, "_quality_radios", []):
            r.configure(state=st)
        for sl, _v in self._pause_vars.values():
            sl.configure(state=st)
        self.sound_chk.configure(state=st)
        if busy and self._live_kind != "preview":
            self._live_stop()                 # книга важнее прослушивания
        self._live_buttons()                  # «Прочитать», «Прослушать» - не во время счёта
        self._pause_flag.clear()
        self.pause_btn.configure(text="Пауза", state="disabled")
        T.restyle(self.pause_btn, "default", "pause")
        if busy:
            self.start_btn.configure(text="Остановить")
            T.restyle(self.start_btn, "danger", "square")
            self.restart_btn.pack_forget()
            self._stop_flag.clear()
            self._t0 = time.time()
            self._live_rtf = None
            self._synth_t0 = None
            self._step = 0
            self._tick()
        else:
            self.start_btn.configure(text="Озвучить книгу", state="normal")
            T.restyle(self.start_btn, "primary", "play")
            self.part_label.configure(text="")
            self.part_progress.set(0)
        if status:
            self.status.configure(text=status)

    def on_main_button(self):
        if self.running:
            self.do_stop()
        else:
            self.do_start()

    def _tick(self):
        if self.running:
            now = self._paused_at if self._pause_flag.is_set() else time.time()
            self.timer.configure(text=_fmt_clock(now - self._t0))
            self.after(1000, self._tick)

    # прогресс из конвейера (вызывается из фонового потока)
    def _progress(self, fr: float, msg: str):
        self.ui(lambda: self._show_progress(fr, msg))

    def _show_progress(self, fr: float, msg: str):
        self.progress.set(fr)
        if self._pause_flag.is_set():
            return
        live = self._live if self._live_kind == "preview" else None
        if live is not None and live.plan_items:
            return                            # пробник звучит: статус пишет поток
        if fr >= 0.95 or msg.startswith(("Склеиваю", "Собираю")):
            step, what = 3, "Сборка файлов"
        elif msg.startswith(("Озвучиваю", "Часть")):
            step, what = 2, "Озвучка"
        else:
            step, what = 1, "Подготовка текста: числа и ударения"
        self._step = step
        detail = msg
        left = ""
        if step == 2:
            sfr = max(0.0, min(1.0, (fr - 0.05) / 0.9))
            if self._synth_t0 is None:
                self._synth_t0, self._synth_fr = time.time(), sfr
            elif sfr - self._synth_fr > 0.01:
                spent = time.time() - self._synth_t0
                rest = spent / (sfr - self._synth_fr) * (1 - sfr)
                left = f" · осталось ~{_fmt_hms(rest)}"
            detail = msg.replace("Озвучиваю: ", "").replace("часть", "часть", 1)
            rtf = (f" · RTF {self._live_rtf:.2f}".replace(".", ",")
                   if self._live_rtf else "")
            text = f"Шаг 2 из 3 · Озвучка - {detail}, {int(fr * 100)} %{left}{rtf}"
        elif step == 3:
            text = f"Шаг 3 из 3 · {what}" if fr < 1 else "Готово"
        elif msg.startswith("Скачиваю модели"):
            text = f"Шаг 1 из 3 · {msg}"      # первый запуск: честно, что идёт скачивание
        else:
            text = f"Шаг 1 из 3 · {what}"
        self.status.configure(text=text)

    def _part_progress(self, k: int, m: int, fr: float):
        text = f"Часть {k} из {m} - {int(fr * 100)}%"
        self.ui(lambda: (self.part_progress.set(fr), self.part_label.configure(text=text)))

    def _pause_gate(self):
        """Зовётся синтезом перед каждым куском. Держит, пока стоит пауза."""
        while self._pause_flag.is_set() and not self._stop_flag.is_set():
            time.sleep(0.3)

    def do_pause(self):
        if not self.running:
            return
        if self._live_kind == "preview" and self._live is not None:
            live = self._live                 # пробник: пауза звука, счёт идёт дальше
            if live.paused:
                live.resume()
                self.pause_btn.configure(text="Пауза")
                T.restyle(self.pause_btn, "default", "pause")
            else:
                live.pause()
                self.pause_btn.configure(text="Продолжить")
                T.restyle(self.pause_btn, "primary", "play")
            return
        if self._pause_flag.is_set():                    # продолжить
            self._t0 += time.time() - self._paused_at    # пауза не в счёт
            if self._synth_t0 is not None:
                self._synth_t0 += time.time() - self._paused_at
            self._pause_flag.clear()
            keepawake.resume()
            self.pause_btn.configure(text="Пауза")
            T.restyle(self.pause_btn, "default", "pause")
            self.status.configure(text="Продолжаю…")
        else:
            self._paused_at = time.time()
            self._pause_flag.set()
            keepawake.pause()             # на паузе Mac может спать (пауза «на ночь»)
            self.pause_btn.configure(text="Продолжить")
            T.restyle(self.pause_btn, "primary", "play")
            self.status.configure(text="Пауза - доделываю текущий кусок и жду. Готовое на месте")

    def do_stop(self):
        if self._live_kind == "preview" and self._live is not None:
            self._live.stop()                 # звук - сразу, счёт - на ближайшем стыке
        self._stop_flag.set()
        self._pause_flag.clear()          # иначе стоящий на паузе синтез не узнает
        self.start_btn.configure(state="disabled")
        self.status.configure(text="Останавливаю после текущего куска…")

    def _logln(self, text: str):
        self.ui(lambda: (self.log.insert("end", text + "\n"), self.log.see("end")))

    def _log_settings(self, what: str, engine: str, kw: dict, voice: str):
        fx = load_fx(default_voices_dir(), voice)
        e = self.est
        self.log.insert("end", f"{what}: {os.path.basename(self.book_path)}\n")
        if e is not None:
            self.log.insert("end", f"  книга: {_files(e.chapters)}, ~{_fmt_hms(e.audio_sec)} "
                                   f"звука, язык {getattr(e, 'language', '?')}\n")
        size = "одним файлом" if self.single_file() else f"части по {self.len_var.get()}"
        self.log.insert("end", f"  голос: {voice} · движок «{self.profile().title}» "
                               f"({engine}, {kw})\n  запись: {self.fmt_var.get()} · {size}\n"
                               f"  звучание: темп ×{fx.speed:.2f}, высота {fx.pitch:+.1f}, "
                               f"бас {fx.bass:+.1f}, яркость {fx.brightness:+.1f}\n\n")

    def do_preview(self):
        """«Проверить голос (1 мин)»: звук идёт по мере готовности, а в конце пробник
        ложится файлом рядом с книгой - как раньше."""
        if self.running or not self._ready():
            return
        path, voice = self.book_path, self.current_voice()
        engine, kw = self._eng_choice()
        opts, fmt = self._conv_options(), self.fmt_var.get()
        self._live_stop()
        self._set_busy(True, "Шаг 1 из 3 · Готовлю пробник…")
        live = self._live_new("preview", engine, kw, voice, opts.sound_check and _clone_engine(engine),
                              lambda t: self.status.configure(text="Пробник · " + t))
        self.pause_btn.configure(state="normal")
        self.progress.set(0)
        self._log_settings("Пробник (одна минута)", engine, kw, voice)
        t0 = time.time()

        def work():
            try:
                out = preview_sample(path, engine=engine, eng_kwargs=kw, voice=voice,
                                     minutes=1.0, options=opts, progress=self._progress,
                                     live=live)
                import shutil
                import soundfile as sf
                a, sr = sf.read(out)
                if len(a) / sr < 3.0:
                    raise RuntimeError("синтез не дал звука (меньше 3 секунд). Прежний "
                                       "пробник не тронут - подробности в «Подробностях».")
                ext = FORMATS.get(fmt, FORMATS[DEFAULT_FORMAT])[0]
                out_path = os.path.splitext(path)[0] + " - пробник" + ext
                # с поправками голоса - как в книге (до 3.4 пробник шёл без них)
                polish_wav_to_mp3(out, out_path, fmt=fmt,
                                  fx=load_fx(default_voices_dir(), voice).filter_chain())
                shutil.rmtree(os.path.dirname(out), ignore_errors=True)
                dur = len(a) / sr
                fs = live.first_sound
                self._logln(f"Пробник: {dur:.0f} с звука, счёт {_fmt_hms(time.time() - t0)}"
                            + (f", первый звук через {fs:.1f} с, запинок {live.stalls}"
                               if fs is not None else ""))
                self.ui(lambda: self._add_part("Пробник", out_path, f"{dur:.0f} с звука"))
                if live.sink is None:
                    self._play_plain(a.astype("float32"), sr, "")
                live.wait()                    # «Остановить» обрывает и звук
                self.ui(lambda: self._preview_done(live, "Пробник готов - он в списке, рядом с книгой"))
            except Cancelled:
                self.ui(lambda: self._preview_done(live, "Остановлено"))
            except Exception as e:
                self.ui(lambda: self._preview_done(live, ""))
                self._fail(e)
        threading.Thread(target=work, daemon=True).start()

    def _preview_done(self, live, status: str):
        if self._live is live:
            self._live = None
        self._live_kind = ""
        self._set_busy(False, status)

    def do_start(self):
        if self.running or not self._ready():
            return
        path, voice = self.book_path, self.current_voice()
        engine, kw = self._eng_choice()
        opts = self._conv_options()
        resuming = bool(self._resume)
        self._set_busy(True, "Шаг 1 из 3 · Подготовка текста: числа и ударения")
        self.pause_btn.configure(state="normal")
        self.progress.set(0)
        self.log.delete("1.0", "end")
        self._log_settings("Озвучка книги", engine, kw, voice)
        if not resuming:
            self._clear_parts()
        else:
            self._clear_parts()           # части с прошлого раза придут заново, по порядку
        t0 = time.time()

        def on_chapter(ci, title, mp3):
            name = os.path.splitext(os.path.basename(mp3))[0] if mp3 else f"Часть {ci}"
            note = title if title and not title.startswith("Часть ") else ""
            self._logln(f"✓ {ci}: {os.path.basename(mp3) if mp3 else title}")
            self.ui(lambda: self._add_part(name, mp3, note))

        def work():
            try:
                res = convert_book(path, engine=engine, eng_kwargs=kw, voice=voice, options=opts,
                                   progress=self._progress, on_chapter=on_chapter,
                                   should_stop=self._stop_flag.is_set,
                                   pause_gate=self._pause_gate, part_progress=self._part_progress,
                                   on_rtf=lambda r: self.ui(lambda: self._got_rtf(r)))
                took = time.time() - t0
                what = "один файл" if res.chapters == 1 else _files(res.chapters)
                self._logln(f"\nГотово. Звука {_fmt_hms(res.duration)}, счёт {_fmt_hms(took)} "
                            f"(на секунду звука {took / max(res.duration, 1):.2f} с)")
                self.ui(lambda: self._learn_book(res.duration, took))
                if res.mp3_dir:
                    self._logln(f"Папка: {res.mp3_dir}")

                def done():
                    self._set_busy(False, f"Готово: {what}, звука {_fmt_hms(res.duration)}")
                    if opts.single_file and res.mp3_dir:
                        self._load_existing_parts()
                    self._check_resume()
                self.ui(done)
            except Cancelled:
                self._logln("Остановлено. Готовые части остались на диске.")
                self.ui(lambda: (self._set_busy(False, "Остановлено - готовое на месте"),
                                 self._check_resume()))
            except ScannedPdfError as e:
                self._fail(e, "PDF без текстового слоя (скан)")
            except ReaderError as e:
                self._fail(e, "Не удалось прочитать книгу")
            except Exception as e:
                self._fail(e)
        threading.Thread(target=work, daemon=True).start()

    def _on_showlog(self):
        self.cfg["show_log"] = bool(self.showlog_var.get())
        save_config(self.cfg)
        self._details_visible(self.cfg["show_log"])

    def _details_visible(self, on: bool):
        """«Подробности» на главной: только при ошибке или по настройке (3.5).
        Журнал пишется в файл всегда - «Открыть журнал» в Настройках."""
        if on:
            self.details_box.grid()
        else:
            if self._details_open:
                self.toggle_details()
            self.details_box.grid_remove()

    def _fail(self, e, title="Ошибка"):
        log_problem(title, e, show=False)
        self.ui(lambda: self._details_visible(True))
        self.ui(lambda: (self._set_busy(False, "Ошибка - подробности в «Подробностях»"),
                         messagebox.showerror(title, str(e))))

    # ================= журнал =================
    def _enable_copy(self, box):
        """Cmd+C и правая кнопка в журнале - привязки на внутреннем виджете."""
        inner = getattr(box, "_textbox", box)
        import tkinter as tk
        menu = tk.Menu(inner, tearoff=0)
        menu.add_command(label="Скопировать выделенное", command=lambda: self.copy_log(selection=True))
        menu.add_command(label="Скопировать всё", command=lambda: self.copy_log())
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: inner.tag_add("sel", "1.0", "end"))

        def popup(event):
            menu.tk_popup(event.x_root, event.y_root)
            return "break"
        for seq in ("<Command-c>", "<Control-c>", "<Command-C>"):
            inner.bind(seq, lambda e: self.copy_log(selection=True))
        inner.bind("<Command-a>", lambda e: (inner.tag_add("sel", "1.0", "end"), "break")[1])
        for seq in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            inner.bind(seq, popup)

    def copy_log(self, selection: bool = False):
        inner = getattr(self.log, "_textbox", self.log)
        try:
            text = (inner.get("sel.first", "sel.last") if selection
                    else inner.get("1.0", "end").strip())
        except Exception:
            text = inner.get("1.0", "end").strip()      # выделения нет - берём всё
        if not text:
            return "break"
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_btn.configure(text="Скопировано")
        self.after(1600, lambda: self.copy_btn.configure(text="Копировать"))
        return "break"

    class _StderrToLog:
        """Строки движков из stderr - в «Подробности» (через очередь окна)."""

        def __init__(self, app, real):
            self.app, self.real = app, real
            self._buf = ""

        def write(self, chunk):
            self.real.write(chunk)
            self._buf += chunk
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip() and not line.startswith("\r"):
                    # «ℹ …» - справка самой программы - в «Подробности»; всё прочее
                    # в stderr пишут библиотеки (предупреждения, полоски загрузки) -
                    # только в файл журнала (3.5: SyntaxWarning от пакета Silero)
                    t = line.strip()
                    if t.startswith("ℹ "):
                        self.app._logln(t[2:])
                    elif "frames/s" not in t and "it/s]" not in t:
                        log_info("stderr: " + t[:500])

        def flush(self):
            self.real.flush()


def main():
    osutil.fix_streams()               # у оконной сборки Windows нет консоли
    osutil.setup_path()                # ffmpeg из установщика, Homebrew на Mac
    if osutil.IS_WIN:
        try:
            from core import cuda_pack
            cuda_pack.activate()       # ускорение NVIDIA, если докачано
        except Exception as e:
            log_problem("ускорение NVIDIA не подключилось", e, show=False)
    T.apply()
    cfg = load_config()
    ctk.set_appearance_mode(cfg.get("theme", "system"))
    import warnings
    # предупреждения библиотек (SyntaxWarning, DeprecationWarning, UserWarning…) -
    # в файл журнала, не в окно
    warnings.showwarning = lambda m, c, f, ln, file=None, line=None: log_info(
        f"{c.__name__}: {m} ({os.path.basename(str(f))}:{ln})")
    app = App()
    sys.stderr = App._StderrToLog(app, sys.stderr)
    app.mainloop()
    # Выход без разбора объектов Metal (MLX) и onnxruntime: изредка он падал с
    # «recursive_mutex lock failed» (опыт Scribe 2.4)
    os._exit(0)


if __name__ == "__main__":
    main()
