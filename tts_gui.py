#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Herald — окно приложения.

Запуск: двойной щелчок по «Herald.app» (или по «Запустить.command», или
`python tts_gui.py` из активированного .venv).

Настроек намеренно мало: всё, что проверено как бесполезное на этой машине,
из окна убрано (см. README, раздел «Скорость»). Остаётся книга, голос, темп,
длина файла и качество записи. Остальное выставлено по замерам.

Начитка книги идёт часами, поэтому в окне есть оценка времени ДО старта,
всегда видимый таймер, кнопка «Стоп» и потоковая выдача готовых файлов.
"""

from __future__ import annotations

import os
# валидный PYTHONHASHSEED до тяжёлых импортов (иначе падают дочерние процессы)
_hs = os.environ.get("PYTHONHASHSEED")
if _hs is None or not _hs.isdigit():
    os.environ["PYTHONHASHSEED"] = "0"

import time
import threading
import subprocess
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import customtkinter as ctk
import tkinterdnd2
from tkinter import filedialog, messagebox

from core import VERSION, APP_NAME, APP_TAGLINE, ABOUT
from core.assemble import FORMATS, DEFAULT_FORMAT, polish_wav_to_mp3, HD_FILTER
from core.engines import get_engine, list_engines
from core.engines.f5 import default_voices_dir
from core.pipeline import (convert_book, preview_sample, estimate_book,
                           ConvertOptions, Cancelled)
from core.readers import ScannedPdfError, ReaderError
from core.voicefx import VoiceFX, load as load_fx, save as save_fx

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Палитра: тёплый янтарь на графите — вечер и книга, а не офисная таблица.
BG, CARD, CARD2, LINE = "#16181D", "#1E2128", "#252932", "#2E333D"
TEXT, DIM = "#E8EAF0", "#8A92A6"
ACCENT, ACCENT_HOVER, INK = "#E0A458", "#C98F44", "#1A1205"

BOOK_EXT = (".txt", ".md", ".docx", ".epub", ".fb2", ".zip", ".pdf")

# Готовые режимы вместо числа шагов: подписи с замерами на M4 (см. README).
# RTF — секунды расчёта на секунду звука; по нему считаем прогноз времени.
MODES: list[tuple[str, dict, float]] = [
    ("Обычное — рекомендуется", {"steps": 7,  "schedule": "epss"},    0.80),
    ("Быстрее",                 {"steps": 6,  "schedule": "epss"},    0.72),
    ("Точнее, вдвое дольше",    {"steps": 16, "schedule": "uniform"}, 1.70),
]
MODE_TITLES = [m[0] for m in MODES]

# Множитель к темпу, который движок уже подогнал под образец (target_bps в
# LANG_MODELS). «Обычный» = 1.0 и даёт ту скорость чтения, что была одобрена на
# слух, — причём одинаковую для любого голоса, хоть быстрого, хоть медленного.
# Заголовки читаются ещё размереннее (HEADING_SPEED_MUL в pipeline).
ENGINE_TITLES = dict(list_engines())
ENGINE_IDS = [e for e in ("f5mlx", "apple") if e in ENGINE_TITLES]
ENGINE_SHORT = {"f5mlx": "Голос диктора (клон)", "apple": "Системный голос macOS"}
CLONING = ("f5mlx", "f5")            # движки, где голос берётся из образца

FILE_LENGTHS = ["10 минут", "15 минут", "20 минут", "30 минут", "45 минут"]


def _open_path(path: str):
    subprocess.run(["open", path], check=False)


def _fmt_hms(sec: float) -> str:
    """«1 ч 05 мин», «3 мин 56 с», «42 с» — с секундами, пока они заметны."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return f"{h} ч {m:02d} мин"
    if m:
        return f"{m} мин {s:02d} с"
    return f"{s} с"


def _card(parent, **kw) -> ctk.CTkFrame:
    return ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14, **kw)


def _label(parent, text, size=13, color=TEXT, weight="normal", **kw):
    return ctk.CTkLabel(parent, text=text, text_color=color, anchor="w",
                        font=ctk.CTkFont(size=size, weight=weight), **kw)


def _menu(parent, values, variable=None, command=None):
    return ctk.CTkOptionMenu(
        parent, values=values, variable=variable, command=command,
        fg_color=CARD2, button_color=CARD2, button_hover_color=LINE,
        text_color=TEXT, dropdown_fg_color=CARD2, dropdown_text_color=TEXT,
        dropdown_hover_color=LINE, corner_radius=8, height=34,
        font=ctk.CTkFont(size=13), dropdown_font=ctk.CTkFont(size=13))


class App(ctk.CTk, tkinterdnd2.TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        # включаем перетаскивание файлов в окно customtkinter
        self.TkdndVersion = tkinterdnd2.TkinterDnD._require(self)

        self.title(APP_NAME)
        self.geometry("840x800")
        self.minsize(780, 740)
        self.configure(fg_color=BG)

        self.load_sample_btn = None       # появляется в окне настроек
        self._setting_menus: list = []     # блокируются на время работы
        self.voices_list = None
        self.book_path: str | None = None
        self.est = None
        self.running = False
        self._stop_flag = threading.Event()
        self._t0 = 0.0

        self._build_menu()
        self._build_header()
        self._build_drop()
        self._build_settings()
        self._build_actions()
        self._build_progress()
        self.refresh_voices()

    # --- части окна -------------------------------------------------------
    def _build_menu(self):
        """Меню «Правка» с Copy/Select All.

        Без него на macOS Cmd+C в окне не работает вовсе: Tk отправляет
        виртуальное событие <<Copy>> только когда в приложении есть
        соответствующий пункт меню.
        """
        import tkinter as tk
        menubar = tk.Menu(self)
        edit = tk.Menu(menubar, tearoff=0)
        edit.add_command(label="Скопировать", accelerator="Cmd+C",
                         command=lambda: self._menu_event("<<Copy>>"))
        edit.add_command(label="Выделить всё", accelerator="Cmd+A",
                         command=lambda: self._menu_event("<<SelectAll>>"))
        edit.add_separator()
        edit.add_command(label="Скопировать весь отчёт",
                         command=lambda: self.copy_log())
        menubar.add_cascade(label="Правка", menu=edit)
        self.configure(menu=menubar)

    def _menu_event(self, event: str):
        w = self.focus_get()
        if w is not None:
            w.event_generate(event)

    def _build_header(self):
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=26, pady=(22, 4))
        _label(head, APP_NAME, size=30, weight="bold").pack(side="left")
        _label(head, f"   {APP_TAGLINE}", size=13, color=DIM).pack(side="left", pady=(12, 0))
        _label(head, VERSION, size=12, color=DIM).pack(side="right", pady=(12, 0))

    def _build_drop(self):
        self.drop = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14,
                                 border_width=2, border_color=LINE, height=128)
        self.drop.pack(fill="x", padx=26, pady=8)
        self.drop.pack_propagate(False)
        self.drop_label = ctk.CTkLabel(
            self.drop, justify="center", text_color=TEXT,
            font=ctk.CTkFont(size=17, weight="bold"), text="Перетащи книгу сюда")
        self.drop_label.pack(expand=True, pady=(18, 0))
        self.drop_hint = ctk.CTkLabel(self.drop, text="txt · md · docx · epub · fb2 · pdf",
                                      text_color=DIM, font=ctk.CTkFont(size=12))
        self.drop_hint.pack()
        row = ctk.CTkFrame(self.drop, fg_color="transparent")
        row.pack(pady=(8, 16))
        ctk.CTkButton(row, text="Выбрать файл", width=130, height=30, corner_radius=8,
                      fg_color=CARD2, hover_color=LINE, text_color=TEXT,
                      command=self.choose_book).pack(side="left", padx=4)
        self.clear_btn = ctk.CTkButton(row, text="Очистить", width=100, height=30,
                                       corner_radius=8, fg_color="transparent",
                                       hover_color=LINE, text_color=DIM,
                                       border_width=1, border_color=LINE,
                                       command=self.clear_book)
        self.clear_btn.pack(side="left", padx=4)
        for w in (self.drop, self.drop_label, self.drop_hint):
            w.drop_target_register(tkinterdnd2.DND_FILES)
            w.dnd_bind("<<Drop>>", self.on_drop)

    def _build_settings(self):
        card = _card(self)
        card.pack(fill="x", padx=26, pady=8)
        for c in (0, 1, 2):
            card.grid_columnconfigure(c, weight=1, uniform="s")

        def cell(row, col, title, factory):
            _label(card, title, size=12, color=DIM).grid(
                row=row, column=col, sticky="w", padx=14, pady=(14, 3))
            w = factory()
            w.grid(row=row + 1, column=col, sticky="ew", padx=14)
            self._setting_menus.append(w)
            return w

        self.engine_var = ctk.StringVar(value=ENGINE_IDS[0])
        cell(0, 0, "Чем озвучивать",
             lambda: _menu(card, [ENGINE_SHORT[e] for e in ENGINE_IDS],
                           command=self.on_engine))
        # голосу отдаём две колонки: имена дикторов длинные, и в одной клетке
        # меню то растягивалось, то поджималось, сдвигая всё вокруг
        self.voice_var = ctk.StringVar(value="")
        voice_box = ctk.CTkFrame(card, fg_color="transparent")
        _label(card, "Голос", size=12, color=DIM).grid(
            row=0, column=1, columnspan=2, sticky="w", padx=14, pady=(14, 3))
        voice_box.grid(row=1, column=1, columnspan=2, sticky="ew", padx=14)
        # сетка, а не pack: при длинном имени голоса меню раздувалось и
        # выдавливало кнопку за край — она то меняла размер, то исчезала
        voice_box.grid_columnconfigure(0, weight=1)
        voice_box.grid_columnconfigure(1, weight=0, minsize=40)
        self.voice_menu = _menu(voice_box, [""], variable=self.voice_var)
        self.voice_menu.grid(row=0, column=0, sticky="ew")
        self._setting_menus.append(self.voice_menu)
        # правим звучание ровно того голоса, который выбран рядом
        self.fx_btn = ctk.CTkButton(
            voice_box, text="♪", width=34, height=34, corner_radius=8,
            fg_color=CARD2, hover_color=LINE, text_color=DIM,
            font=ctk.CTkFont(size=15), command=self.show_voice_fx)
        self.fx_btn.grid(row=0, column=1, sticky="e", padx=(6, 0))
        self.mode_var = ctk.StringVar(value=MODE_TITLES[0])
        self.mode_menu = cell(2, 0, "Качество синтеза",
                              lambda: _menu(card, MODE_TITLES, variable=self.mode_var,
                                            command=lambda *_: self.show_estimate()))
        self.len_var = ctk.StringVar(value="15 минут")
        cell(2, 1, "Длина одного файла",
             lambda: _menu(card, FILE_LENGTHS, variable=self.len_var,
                           command=lambda *_: self.reestimate()))
        self.fmt_var = ctk.StringVar(value=DEFAULT_FORMAT)
        cell(2, 2, "Качество записи",
             lambda: _menu(card, list(FORMATS), variable=self.fmt_var,
                           command=lambda *_: self.show_estimate()))

        self.est_label = ctk.CTkLabel(card, text="Книга не выбрана", anchor="w",
                                      justify="left", text_color=DIM,
                                      font=ctk.CTkFont(size=12))
        self.est_label.grid(row=4, column=0, columnspan=3, sticky="ew",
                            padx=14, pady=(16, 14))

    def _build_actions(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=26, pady=(4, 8))
        self.start_btn = ctk.CTkButton(
            row, text="Озвучить книгу", height=44, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=INK,
            font=ctk.CTkFont(size=15, weight="bold"), command=self.do_start)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.preview_btn = ctk.CTkButton(
            row, text="Послушать минуту", width=170, height=44, corner_radius=10,
            fg_color=CARD2, hover_color=LINE, text_color=TEXT,
            font=ctk.CTkFont(size=13), command=self.do_preview)
        self.preview_btn.pack(side="left", padx=8)
        self.stop_btn = ctk.CTkButton(
            row, text="Стоп", width=84, height=44, corner_radius=10, state="disabled",
            fg_color="transparent", hover_color=LINE, border_width=1,
            border_color=LINE, text_color=DIM, command=self.do_stop)
        self.stop_btn.pack(side="left")
        ctk.CTkButton(row, text="⚙", width=44, height=44, corner_radius=10,
                      fg_color="transparent", hover_color=LINE, border_width=1,
                      border_color=LINE, text_color=DIM,
                      font=ctk.CTkFont(size=16), command=self.show_settings).pack(
            side="left", padx=(8, 0))

    def _build_progress(self):
        card = _card(self)
        card.pack(fill="both", expand=True, padx=26, pady=(4, 22))
        bar = ctk.CTkFrame(card, fg_color="transparent")
        bar.pack(fill="x", padx=16, pady=(16, 6))
        self.status = ctk.CTkLabel(bar, text="Готово к работе", anchor="w",
                                   text_color=TEXT, font=ctk.CTkFont(size=13))
        self.status.pack(side="left")
        # Таймер — отдельной надписью. Раньше он дописывался к тексту статуса и
        # пропадал каждый раз, когда статус обновлялся: казалось, что мигает.
        self.timer = ctk.CTkLabel(bar, text="0:00", text_color=DIM,
                                  font=ctk.CTkFont(size=13))
        self.timer.pack(side="right")
        self.progress = ctk.CTkProgressBar(card, height=6, corner_radius=3,
                                           progress_color=ACCENT, fg_color=CARD2)
        self.progress.pack(fill="x", padx=16)
        self.progress.set(0)
        self.log = ctk.CTkTextbox(card, fg_color=CARD2, text_color=TEXT,
                                  corner_radius=10, font=ctk.CTkFont(size=12))
        self.log.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        self._enable_copy(self.log)
        foot = ctk.CTkFrame(card, fg_color="transparent")
        foot.pack(fill="x", padx=16, pady=(0, 14))
        self.copy_btn = ctk.CTkButton(
            foot, text="Скопировать отчёт", width=170, height=30, corner_radius=8,
            fg_color=CARD2, hover_color=LINE, text_color=DIM,
            font=ctk.CTkFont(size=12), command=self.copy_log)
        self.copy_btn.pack(side="right")


    # --- книга -----------------------------------------------------------
    def on_drop(self, event):
        # tk отдаёт путь в фигурных скобках, если в нём есть пробелы
        paths = [p for p in self.tk.splitlist(event.data)
                 if p.lower().endswith(BOOK_EXT)]
        if not paths:
            messagebox.showwarning("Не та книга",
                                   "Поддерживаются: " + ", ".join(BOOK_EXT))
            return
        self._set_book(paths[0])

    def choose_book(self):
        p = filedialog.askopenfilename(
            title="Выберите книгу",
            filetypes=[("Книги", "*.epub *.fb2 *.txt *.md *.docx *.pdf"),
                       ("Все файлы", "*.*")])
        if p:
            self._set_book(p)

    def clear_book(self):
        if self.running:
            return
        self.book_path = None
        self.est = None
        self.drop_label.configure(text="Перетащи книгу сюда")
        self.drop_hint.configure(text="txt · md · docx · epub · fb2 · pdf")
        self.est_label.configure(text="Книга не выбрана")
        self.progress.set(0)
        self.log.delete("1.0", "end")
        self.status.configure(text="Готово к работе")
        self.timer.configure(text="0:00")

    def _set_book(self, p: str):
        self.book_path = p
        self.drop_label.configure(text=os.path.basename(p))
        self.drop_hint.configure(text=os.path.dirname(p))
        self.reestimate()

    def reestimate(self):
        if not self.book_path:
            return
        self.est = None
        self.est_label.configure(text="Считаю объём книги…")
        minutes = self.file_minutes()

        def work():
            try:
                est = estimate_book(self.book_path, minutes)
                self.after(0, lambda: (setattr(self, "est", est), self.show_estimate()))
            except Exception as e:
                self.after(0, lambda: self.est_label.configure(
                    text=f"Не удалось прочитать книгу: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def file_minutes(self) -> float:
        return float(self.len_var.get().split()[0])

    def show_estimate(self):
        if not self.est:
            return
        e = self.est
        if self.engine_var.get() not in CLONING:
            rtf = 0.05
        else:
            rtf = dict(zip(MODE_TITLES, (m[2] for m in MODES)))[self.mode_var.get()]
        # темп голоса меняет длину звука: медленнее читает — дольше звучит
        rate = load_fx(default_voices_dir(), self.current_voice()).speed or 1.0
        audio_sec = e.audio_sec / rate
        kbps = {"mp3 192 кбит/с (обычный)": 192, "mp3 320 кбит/с": 320}.get(
            self.fmt_var.get(), 750)
        mb = audio_sec * kbps / 8 / 1024
        size = f"{mb / 1024:.1f} ГБ" if mb >= 1024 else f"{mb:.0f} МБ"
        lang = {"ru": "русская книга", "en": "английская книга"}.get(
            getattr(e, "language", "ru"), "")
        self.est_label.configure(
            text=(f"{lang} · {e.chapters} файлов · {_fmt_hms(audio_sec)} звука · "
                  f"расчёт около {_fmt_hms(audio_sec * rtf)} · на диске ~{size}"))

    # --- движок и голоса --------------------------------------------------
    def on_engine(self, title: str):
        for eid in ENGINE_IDS:
            if ENGINE_SHORT[eid] == title:
                self.engine_var.set(eid)
                break
        clone = self.engine_var.get() in CLONING
        self.mode_menu.configure(state="normal" if clone else "disabled")
        self.refresh_voices()
        self.show_estimate()

    # В меню голос показывается с пометкой языка («Игорь Ященко · русский»),
    # а движку передаётся чистое имя — метка тут только для человека.
    LANG_MARK = {"ru": "русский", "en": "английский"}

    def refresh_voices(self):
        try:
            self._voices = get_engine(self.engine_var.get()).list_voices()
        except Exception:
            self._voices = []
        book_lang = getattr(self.est, "language", None)
        # голоса нужного языка — наверх, чужие всё равно оставляем в списке
        if book_lang:
            self._voices.sort(key=lambda v: v.language != book_lang)
        labels = [self._voice_label(v) for v in self._voices] or [""]
        self.voice_menu.configure(values=labels)
        if self.voice_var.get() not in labels:
            self.voice_var.set(labels[0])
        self.voice_var.trace_add("write", lambda *_: self.show_estimate()) \
            if not getattr(self, "_voice_traced", False) else None
        self._voice_traced = True

    def _select_voice(self, vid: str):
        for v in getattr(self, "_voices", []):
            if v.id == vid:
                self.voice_var.set(self._voice_label(v))
                return

    VOICE_NAME_LIMIT = 22

    def _voice_label(self, v) -> str:
        # длинные имена укорачиваем: от них меню меняло ширину, и вся строка
        # настроек прыгала влево-вправо при каждом переключении голоса
        name = v.id if len(v.id) <= self.VOICE_NAME_LIMIT else \
            v.id[:self.VOICE_NAME_LIMIT - 1] + "…"
        return f"{name} · {self.LANG_MARK.get(v.language, v.language)}"

    def current_voice(self) -> str:
        """Имя голоса без пометки языка."""
        label = self.voice_var.get()
        for v in getattr(self, "_voices", []):
            if self._voice_label(v) == label:
                return v.id
        return label.split(" · ")[0]

    def load_sample(self):
        messagebox.showinfo(
            "Новый голос",
            "Сейчас нужно выбрать АУДИОФАЙЛ, где читает тот диктор, чьим голосом\n"
            "ты хочешь слушать книги. Подойдёт любая его аудиокнига целиком.\n\n"
            "Дальше всё делается само — ровно так же, как был сделан готовый голос:\n"
            "  • программа слушает запись в нескольких местах, начиная с начала:\n"
            "    там диктор читает ровнее, а фон чище всего;\n"
            "  • выбирает отрывок 8–12 секунд с тихим фоном и беглой речью;\n"
            "  • разбирает, что там сказано (Whisper large-v3-turbo);\n"
            "  • запоминает голос под тем именем, которое ты дашь.\n\n"
            "Занимает около минуты и делается один раз.")
        p = filedialog.askopenfilename(
            title="Аудиофайл с голосом диктора (например, его аудиокнига)",
            filetypes=[("Аудио", "*.mp3 *.wav *.m4a *.flac *.ogg"), ("Все файлы", "*.*")])
        if not p:
            return
        name = ctk.CTkInputDialog(title="Имя голоса",
                                  text="Как назвать этот голос?").get_input()
        if not name or not name.strip():
            return
        self._set_busy(True, "Слушаю запись и выбираю отрывок…")

        def work():
            try:
                vid = get_engine("f5mlx").prepare_reference(p, name=name.strip())
                self.after(0, lambda: (self.refresh_voices(),
                                       self._select_voice(vid),
                                       self._refresh_voices_list(),
                                       self._set_busy(False, f"Голос готов: {vid}")))
            except Exception as e:
                self._fail(e, "Не удалось подготовить голос")
        threading.Thread(target=work, daemon=True).start()

    # --- настройки --------------------------------------------------------
    def show_settings(self):
        win = ctk.CTkToplevel(self)
        win.title("Настройки")
        win.geometry("700x660")
        win.configure(fg_color=BG)
        win.transient(self)

        top = _card(win)
        top.pack(fill="x", padx=18, pady=(18, 8))
        _label(top, "Голоса", size=16, weight="bold").pack(fill="x", padx=16, pady=(14, 2))
        _label(top, "Голос делается из любой записи нужного диктора — например,\n"
                    "из его аудиокниги. Программа сама найдёт подходящий отрывок.",
               size=12, color=DIM, justify="left").pack(fill="x", padx=16, pady=(0, 10))
        # голоса — столбиком: в одну строку они не помещаются, а список растёт
        self.voices_list = ctk.CTkTextbox(top, height=92, fg_color=CARD2,
                                          text_color=TEXT, corner_radius=8,
                                          font=ctk.CTkFont(size=13),
                                          activate_scrollbars=True)
        self.voices_list.pack(fill="x", padx=16)
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=14)
        self.load_sample_btn = ctk.CTkButton(
            row, text="Добавить голос диктора", height=36, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=INK,
            command=self.load_sample)
        self.load_sample_btn.pack(side="left")
        ctk.CTkButton(row, text="Папка с голосами", height=36, corner_radius=8,
                      fg_color=CARD2, hover_color=LINE, text_color=TEXT,
                      command=lambda: _open_path(default_voices_dir())).pack(
            side="left", padx=8)
        self._refresh_voices_list()

        box = ctk.CTkTextbox(win, wrap="word", fg_color=CARD, text_color=TEXT,
                             corner_radius=12, font=ctk.CTkFont(size=12))
        box.pack(fill="both", expand=True, padx=18, pady=8)
        box.insert("1.0", ABOUT)
        box.configure(state="disabled")
        ctk.CTkButton(win, text="Закрыть", width=110, height=36, corner_radius=8,
                      fg_color=CARD2, hover_color=LINE, text_color=TEXT,
                      command=win.destroy).pack(pady=(0, 18))
        win.after(120, win.lift)

    # --- ручная подстройка голоса ----------------------------------------
    FX_PHRASE = {
        "ru": "Дом сто+ял на краю дер+евни, и к+аждое +утро над ним подним+ался д+ым.",
        "en": "The house stood at the edge of the village, and smoke rose above it every morning.",
    }

    def show_voice_fx(self):
        """Высота, бас и яркость выбранного голоса — со слуховой проверкой.

        Характер речи менять нельзя: интонация и манера приходят из образца.
        А вот подправить голос как на пульте — можно, и слышно это сразу:
        фраза синтезируется ОДИН раз, дальше ползунки меняют только обработку
        готового звука, а это уже сотые доли секунды.
        """
        vid = self.current_voice()
        if not vid:
            messagebox.showwarning("Нет голоса", "Сначала добавь голос диктора.")
            return
        fx = load_fx(default_voices_dir(), vid)
        self._fx_raw = None                      # синтезированная фраза (wav)

        win = ctk.CTkToplevel(self)
        win.title(f"Звучание · {vid}")
        win.geometry("580x500")
        win.configure(fg_color=BG)
        win.transient(self)

        card = _card(win)
        card.pack(fill="both", expand=True, padx=18, pady=18)
        _label(card, f"Голос «{vid}»", size=16, weight="bold").pack(
            fill="x", padx=18, pady=(16, 2))
        _label(card, "Настройки запоминаются для этого голоса и применяются\n"
                     "при каждой озвучке. Интонацию и манеру так не поменять —\n"
                     "они приходят из образца.",
               size=12, color=DIM, justify="left").pack(fill="x", padx=18, pady=(0, 10))

        rows = [("Темп", "speed", 0.8, 1.25, "×"),
                ("Высота", "pitch", -4, 4, " полутона"),
                ("Бас", "bass", -6, 6, " дБ"),
                ("Яркость", "brightness", -6, 6, " дБ")]
        self._fx_vars = {}
        for title, key, lo, hi, unit in rows:
            line = ctk.CTkFrame(card, fg_color="transparent")
            line.pack(fill="x", padx=18, pady=6)
            _label(line, title, size=13, width=80).pack(side="left")
            val = ctk.CTkLabel(line, text="", width=90, text_color=DIM,
                               font=ctk.CTkFont(size=12))
            val.pack(side="right")
            steps = 18 if key == "speed" else int((hi - lo) * 4)
            sl = ctk.CTkSlider(line, from_=lo, to=hi, number_of_steps=steps,
                               button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                               progress_color=ACCENT, fg_color=CARD2)
            sl.pack(side="left", fill="x", expand=True, padx=10)
            sl.set(getattr(fx, key))
            self._fx_vars[key] = (sl, val, unit)
            sl.configure(command=lambda _v, k=key: self._fx_slider(k))
            self._fx_changed(key)

        btns = ctk.CTkFrame(card, fg_color="transparent")
        btns.pack(fill="x", padx=18, pady=(14, 16))
        self.fx_play_btn = ctk.CTkButton(
            btns, text="Прослушать", height=36, corner_radius=8, fg_color=ACCENT,
            hover_color=ACCENT_HOVER, text_color=INK, command=self._fx_play)
        self.fx_play_btn.pack(side="left")
        ctk.CTkButton(btns, text="Сбросить", height=36, width=110, corner_radius=8,
                      fg_color=CARD2, hover_color=LINE, text_color=TEXT,
                      command=self._fx_reset).pack(side="left", padx=8)
        ctk.CTkButton(btns, text="Сохранить", height=36, width=120, corner_radius=8,
                      fg_color=CARD2, hover_color=LINE, text_color=TEXT,
                      command=lambda: self._fx_save(vid, win)).pack(side="right")
        win.after(120, win.lift)

    def _fx_current(self) -> VoiceFX:
        vals = {}
        for k, (sl, _l, _u) in self._fx_vars.items():
            vals[k] = round(sl.get(), 2 if k == "speed" else 1)
        return VoiceFX(**vals)

    def _fx_slider(self, key: str):
        self._fx_changed(key)
        if key == "speed":
            self._fx_raw = None      # темп задаётся при синтезе, фразу пересчитаем

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
        """Синтезируем фразу один раз, потом только пересобираем обработку."""
        vid = self.current_voice()
        fx = self._fx_current()
        self.fx_play_btn.configure(state="disabled", text="Считаю…")

        def work():
            try:
                import tempfile
                if self._fx_raw is None:
                    lang = next((v.language for v in self._voices if v.id == vid), "ru")
                    eng = get_engine("f5mlx", language=lang, speed=fx.speed)
                    audio = eng.synth_chunk(self.FX_PHRASE.get(lang, self.FX_PHRASE["ru"]),
                                            vid)
                    import soundfile as sf
                    raw = os.path.join(tempfile.mkdtemp(prefix="herald_fx_"), "raw.wav")
                    sf.write(raw, audio, 24000)
                    self._fx_raw = raw
                out = os.path.join(os.path.dirname(self._fx_raw), "play.wav")
                chain = ",".join(x for x in (HD_FILTER, fx.filter_chain()) if x)
                subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                "-i", self._fx_raw, "-af", chain, out], check=True)
                subprocess.run(["afplay", out], check=False)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
            finally:
                # окно настройки могли закрыть, пока считалась фраза
                self.after(0, self._fx_button_ready)
        threading.Thread(target=work, daemon=True).start()

    def _fx_button_ready(self):
        btn = getattr(self, "fx_play_btn", None)
        try:
            if btn is not None and btn.winfo_exists():
                btn.configure(state="normal", text="Прослушать")
        except Exception:
            pass          # виджет уже уничтожен — ничего страшного

    def _fx_save(self, vid: str, win):
        save_fx(default_voices_dir(), vid, self._fx_current())
        self.status.configure(text=f"Настройки голоса «{vid}» сохранены")
        self._refresh_voices_list()
        self.show_estimate()          # темп влияет на длину и прогноз
        win.destroy()

    def _refresh_voices_list(self):
        if self.voices_list is None or not self.voices_list.winfo_exists():
            return
        try:
            voices = get_engine("f5mlx").list_voices()
        except Exception:
            voices = []
        fxdir = default_voices_dir()
        lines = []
        for v in voices:
            fx = load_fx(fxdir, v.id)
            mark = "" if fx.is_neutral() else "   (звучание настроено)"
            lines.append(f"•  {self._voice_label(v)}{mark}")
        self.voices_list.configure(state="normal")
        self.voices_list.delete("1.0", "end")
        self.voices_list.insert("1.0", "\n".join(lines) or "пока ни одного")

    # --- общее -----------------------------------------------------------
    # Осторожно с именами методов и полей: tkinter.Misc уже занимает `_options`,
    # `_configure`, `_w` и другие. Свой метод `_options()` молча ломает окно на
    # старте (TypeError в configure), поэтому здесь `_conv_options`.
    def _eng_kwargs(self) -> dict:
        kw: dict = {}
        if self.engine_var.get() in CLONING:
            kw.update(dict(zip(MODE_TITLES, (m[1] for m in MODES)))[self.mode_var.get()])
        return kw

    def _conv_options(self) -> ConvertOptions:
        return ConvertOptions(make_m4b=False, make_chapter_mp3=True,
                              minutes_per_file=self.file_minutes(),
                              audio_format=self.fmt_var.get())

    def _set_busy(self, busy: bool, status: str = ""):
        self.running = busy
        state = "disabled" if busy else "normal"
        # во время счёта оставляем живой только «Стоп»: любое переключение
        # настроек на ходу всё равно ни на что не повлияет, а нажатая по
        # ошибке кнопка может запустить второй синтез поверх первого
        for b in (self.start_btn, self.preview_btn, self.clear_btn, self.fx_btn):
            b.configure(state=state)
        for w in self._setting_menus:
            w.configure(state=state)
        self.stop_btn.configure(state="normal" if busy else "disabled",
                                text_color=TEXT if busy else DIM)
        if getattr(self, "load_sample_btn", None) is not None:
            try:
                self.load_sample_btn.configure(state=state)
            except Exception:
                pass                      # окно настроек могли уже закрыть
        if status:
            self.status.configure(text=status)
        if busy:
            self._stop_flag.clear()
            self._t0 = time.time()
            self._tick()

    def _tick(self):
        if self.running:
            el = int(time.time() - self._t0)
            self.timer.configure(text=f"{el // 60}:{el % 60:02d}")
            self.after(1000, self._tick)

    def _progress(self, fr: float, msg: str):
        self.after(0, lambda: (self.progress.set(fr), self.status.configure(text=msg)))

    def _logln(self, text: str):
        self.after(0, lambda: (self.log.insert("end", text + "\n"), self.log.see("end")))

    def _enable_copy(self, box):
        """Копирование из лога: Cmd+C, Ctrl+C и правая кнопка мыши.

        Привязки ставим на ВНУТРЕННИЙ tkinter-виджет (box._textbox), а не на
        CTkTextbox: обёртка события клавиш до него не доносит, поэтому Cmd+C
        по выделенному тексту не работал вообще. Плюс своё контекстное меню —
        стандартного у Tk на macOS нет.
        """
        inner = getattr(box, "_textbox", box)
        import tkinter as tk

        menu = tk.Menu(inner, tearoff=0)
        menu.add_command(label="Скопировать выделенное",
                         command=lambda: self.copy_log(selection=True))
        menu.add_command(label="Скопировать всё", command=lambda: self.copy_log())
        menu.add_separator()
        menu.add_command(label="Выделить всё",
                         command=lambda: (inner.tag_add("sel", "1.0", "end"), None))

        def popup(event):
            menu.tk_popup(event.x_root, event.y_root)
            return "break"

        for seq in ("<Command-c>", "<Control-c>", "<Command-C>"):
            inner.bind(seq, lambda e: self.copy_log(selection=True))
        inner.bind("<Command-a>", lambda e: (inner.tag_add("sel", "1.0", "end"), "break")[1])
        for seq in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            inner.bind(seq, popup)

    def copy_log(self, selection: bool = False):
        """Положить отчёт в буфер обмена — его удобно прислать целиком."""
        inner = getattr(self.log, "_textbox", self.log)
        try:
            text = (inner.get("sel.first", "sel.last") if selection
                    else inner.get("1.0", "end").strip())
        except Exception:
            text = inner.get("1.0", "end").strip()      # выделения нет — берём всё
        if not text:
            return "break"
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_btn.configure(text="Скопировано ✓")
        self.after(1600, lambda: self.copy_btn.configure(text="Скопировать отчёт"))
        return "break"

    def _ready(self) -> bool:
        if not self.book_path or not os.path.isfile(self.book_path):
            messagebox.showwarning("Нет книги", "Перетащи книгу в окно.")
            return False
        if not self.current_voice():
            messagebox.showwarning(
                "Нет голоса",
                "Открой настройки (⚙) и нажми «Добавить голос диктора».")
            return False
        return True

    def do_stop(self):
        self._stop_flag.set()
        self.status.configure(text="Останавливаю после текущего куска…")

    # --- пробник ---------------------------------------------------------
    def do_preview(self):
        if self.running or not self._ready():
            return
        self._set_busy(True, "Готовлю пробник…")
        self.progress.set(0)
        self.log.delete("1.0", "end")
        self._log_settings("Пробник (одна минута)")

        def work():
            try:
                out = preview_sample(self.book_path, engine=self.engine_var.get(),
                                     eng_kwargs=self._eng_kwargs(),
                                     voice=self.current_voice(), minutes=1.0,
                                     options=self._conv_options(), progress=self._progress)
                import soundfile as _sf
                _a, _sr = _sf.read(out)
                if len(_a) / _sr < 3.0:
                    raise RuntimeError(
                        "синтез не дал звука (получилось меньше 3 секунд). "
                        "Прежний пробник не тронут — смотри строки с ⚠︎ выше.")
                ext = FORMATS.get(self.fmt_var.get(), FORMATS[DEFAULT_FORMAT])[0]
                out_path = os.path.splitext(self.book_path)[0] + " — пробник" + ext
                # та же полировка, что у глав: иначе пробник звучит иначе, чем книга
                polish_wav_to_mp3(out, out_path, fmt=self.fmt_var.get())
                self._logln(f"Звука {len(_a) / _sr:.0f} c, "
                            f"счёт {_fmt_hms(time.time() - self._t0)}")
                self._logln(f"Пробник готов: {out_path}")
                # сам файл не открываем: слушатель включит его, когда захочет
                self.after(0, lambda: self._set_busy(
                    False, "Пробник готов — файл рядом с книгой"))
            except Exception as e:
                self._fail(e)
        threading.Thread(target=work, daemon=True).start()

    # --- вся книга -------------------------------------------------------
    def _log_settings(self, what: str):
        """Шапка отчёта: с какими настройками считали. Чтобы можно было прислать."""
        vid = self.current_voice()
        fx = load_fx(default_voices_dir(), vid)
        e = self.est
        self._logln(f"{what}: {os.path.basename(self.book_path)}")
        if e is not None:
            self._logln(f"  книга: {e.chapters} файлов, ~{_fmt_hms(e.audio_sec)} звука, "
                        f"язык {getattr(e, 'language', '?')}")
        self._logln(f"  голос: {vid}  ·  {ENGINE_SHORT.get(self.engine_var.get(), '')}")
        self._logln(f"  качество синтеза: {self.mode_var.get()}  ·  "
                    f"запись: {self.fmt_var.get()}  ·  файл по {self.len_var.get()}")
        self._logln(f"  настройки голоса: темп ×{fx.speed:.2f}, высота {fx.pitch:+.1f}, "
                    f"бас {fx.bass:+.1f}, яркость {fx.brightness:+.1f}")
        self._logln("")

    def do_start(self):
        if self.running or not self._ready():
            return
        self._set_busy(True, "Старт…")
        self.progress.set(0)
        self.log.delete("1.0", "end")
        self._log_settings("Озвучка книги")

        def on_chapter(ci, title, mp3):
            self._logln(f"✓ {ci}: {os.path.basename(mp3) if mp3 else title}"
                        f"{'  — можно слушать' if mp3 else ''}")

        def work():
            try:
                res = convert_book(self.book_path, engine=self.engine_var.get(),
                                   eng_kwargs=self._eng_kwargs(),
                                   voice=self.current_voice(),
                                   options=self._conv_options(),
                                   progress=self._progress, on_chapter=on_chapter,
                                   should_stop=self._stop_flag.is_set)
                took = time.time() - self._t0
                mins = int(res.duration // 60)
                self._logln("")
                self._logln(f"Готово. Звука {_fmt_hms(res.duration)}, "
                            f"счёт {_fmt_hms(took)} "
                            f"(на секунду звука {took / max(res.duration, 1):.2f} с)")
                self.after(0, lambda: self._set_busy(
                    False, f"Готово: {res.chapters} файлов, ~{mins} мин звука"))
                if res.mp3_dir:
                    self._logln(f"Папка: {res.mp3_dir}")
                    self.after(0, lambda: _open_path(res.mp3_dir))
            except Cancelled:
                self._logln("Остановлено. Готовые файлы остались на диске.")
                self.after(0, lambda: self._set_busy(False, "Остановлено"))
            except ScannedPdfError as e:
                self._fail(e, "PDF без текстового слоя (скан)")
            except ReaderError as e:
                self._fail(e, "Не удалось прочитать книгу")
            except Exception as e:
                self._fail(e)
        threading.Thread(target=work, daemon=True).start()

    def _fail(self, e, title="Ошибка"):
        traceback.print_exc()
        self.after(0, lambda: (self._set_busy(False, "Ошибка"),
                               messagebox.showerror(title, str(e))))

    class _StderrToLog:
        """Движки пишут «пропущен кусок: …» в stderr, а окно его не показывало.

        Из-за этого сбой синтеза выглядел как «программа молча сделала пустой
        файл». Теперь такие строки видны прямо в окне.
        """

        def __init__(self, app, real):
            self.app, self.real = app, real
            self._buf = ""

        def write(self, chunk):
            self.real.write(chunk)
            self._buf += chunk
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip() and not line.startswith("\r"):
                    self.app._logln("⚠︎ " + line.strip())

        def flush(self):
            self.real.flush()


if __name__ == "__main__":
    app = App()
    import sys as _sys
    _sys.stderr = App._StderrToLog(app, _sys.stderr)
    app.mainloop()
