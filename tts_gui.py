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
from core.i18n import _, tr_msg
from core import i18n
from core.assemble import FORMATS, FORMAT_KBPS, DEFAULT_FORMAT, polish_wav_to_mp3, HD_FILTER
from core.chapters import part_name
from core.engines import get_engine, clone_engine, accel_label
from core.engines.f5 import default_voices_dir
from core.engines.silero import VOICES as SILERO_VOICES
from core.log import log_problem, log_info, LOG_PATH
from core.pipeline import (convert_book, preview_sample, estimate_book, ConvertOptions,
                           Cancelled, resume_info, forget_progress, cleanup_stale_temp,
                           output_dir_for, cached_share, service_size, trash_service,
                           SERVICE_DIR)
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
DEFAULT_SCALE = 1.2        # масштаб по умолчанию - 120 % (стандарт интерфейса, п. 1)
_RESTART = None            # окно пересобирается (язык, меню): что вернуть на место
FAST_VOICE = "aidar"       # «Прочитать вслух» быстрым голосом: готовый голос Silero
SAY_HL_BG, SAY_HL_FG = "#FFE99A", "#1A1A1A"   # подсветка звучащего предложения
SAY_TEMPO = (0.5, 1.5)
CACHED_CHUNK_SEC = 1.0     # кусок из кэша: прочитать, проверить края, склеить     # темп во вкладке (3.6: Андрей хотел медленнее 0,8)
GENDER = {"м": "муж.", "ж": "жен."}
RECENT_MAX = 5             # «Недавние книги» в пустой карточке (задание 12)
SCALE_MIN, SCALE_MAX = 0.8, 2.0    # «Aa» - 80-200 % (Единый стандарт 1.5)
# группы голосов (задание 12): «Мои голоса» - обученные на записях чтеца, потом по языку
GROUPS = (("mine", "Мои голоса"), ("ru", "Русские"), ("en", "Английские"))


# --- мелочи ----------------------------------------------------------------
def _mb(n: int) -> str:
    """Размер по-русски: «747 МБ», «1,2 ГБ»."""
    if n >= 1 << 30:
        return _('{0:.1f} ГБ').format(n / (1 << 30)).replace(".", _dec())
    return _('{0} МБ').format(max(1, round(n / (1 << 20))))


def _kind(v: str) -> str:
    """«Быстрый» / «Клон» - каноническое значение из показанного (RU или EN)."""
    return "Клон" if v in ("Клон", _("Клон")) else "Быстрый"


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
        return _('{0} ч {1:02d} мин').format(h, m)
    if m:
        return _('{0} мин {1:02d} с').format(m, s)
    return _('{0} с').format(s)


def _fmt_clock(sec: float) -> str:
    """Живой таймер: «3 мин 07 с», а за час - «1 ч 05 мин 12 с»."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return _('{0} ч {1:02d} мин {2:02d} с').format(h, m, s) if h else _('{0} мин {1:02d} с').format(m, s)


def _dec() -> str:
    """Десятичный знак языка интерфейса: 0,40 с / 0.40 s."""
    return "," if i18n.lang() == "ru" else "."


def _files(n: int) -> str:
    """«1 файл», «3 файла», «12 файлов» (в английском - 1 file, 21 files)."""
    if i18n.lang() != "ru":
        return f"{n} {_('файл') if n == 1 else _('файлов')}"
    if n % 10 == 1 and n % 100 != 11:
        w = _("файл")
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        w = _("файла")
    else:
        w = _("файлов")
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
    """Голос в списке: обученный, клон (из образца), голос Qwen или готовый."""

    def __init__(self, vid: str, title: str, language: str, ready: bool,
                 engine: str = "silero", gender: str = "", *, mine: bool = False,
                 qwen: bool = False, hidden: bool = False):
        self.id, self.title, self.language, self.ready = vid, title, language, ready
        self.engine, self.gender = engine, gender
        self.mine, self.qwen, self.hidden = mine, qwen, hidden

    def group(self) -> str:
        return "mine" if self.mine else ("en" if self.language == "en" else "ru")

    def label(self) -> str:
        # имена готовых голосов - часть программы (переводятся), клонов - данные
        name = _short(_(self.title), VOICE_NAME_LIMIT)
        tail = " · Qwen" if self.qwen else ""
        return f"{name} · {_(LANG_MARK.get(self.language, self.language))}{tail}"


def _sorted_voices(vs: list) -> list:
    """Порядок голосов (задание 12): «Мои голоса», русские, английские; внутри - по алфавиту."""
    order = {g: i for i, (g, _t) in enumerate(GROUPS)}
    return sorted(vs, key=lambda v: (order.get(v.group(), 9), _(v.title).lower()))


def _grouped_menu(om, groups: list) -> None:
    """Выпадающий список с заголовками групп («Мои голоса · 2»): заголовок - неактивная
    строка. Рисует свой список и тогда, когда customtkinter перестраивает его сам."""
    labels = [lab for _t, labs in groups for lab in labs] or [""]
    om.configure(values=labels)
    menu = om._dropdown_menu

    def build():
        menu.delete(0, "end")
        for title, labs in groups:
            if not labs:
                continue
            if len([g for g in groups if g[1]]) > 1:
                menu.add_command(label=title, state="disabled")
            for lab in labs:
                menu.add_command(label=lab.ljust(menu._min_character_width),
                                 command=lambda v=lab: menu._button_callback(v))
    menu._add_menu_commands = build
    build()


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
        if not path:                    # «не записалась» - с причиной (задание 12)
            note = " · ".join(x for x in (note, _("синтез не дал звука - подробности в журнале "
                                                  "(Настройки › Открыть журнал)")) if x)
        ctk.CTkLabel(self, text="", image=T.icon("audio-lines", T.MUTED, 18)).grid(
            row=0, column=0, rowspan=2, padx=(14, 10), pady=(10, 0), sticky="n")
        T.label(self, _short(title, 60), 14, bold=True, anchor="w").grid(
            row=0, column=1, sticky="ew", pady=(8, 0))
        T.label(self, note, 12, color=T.MUTED if path else T.WARN, anchor="w", justify="left",
                wraplength=420).grid(row=1, column=1, sticky="ew", pady=(0, 8))
        acts = ctk.CTkFrame(self, fg_color="transparent")
        acts.grid(row=0, column=2, rowspan=2, padx=(8, 10))
        if path:
            self.listen_btn = T.button(
                acts, _("Слушать"), "play", height=28, size=13, width=96,
                command=lambda: self.on_listen(self) if self.on_listen else _open_path(path))
            self.listen_btn.pack(side="left", padx=(0, 4))
            # одна кнопка папки на часть (3.2): «Показать в Finder» рядом с плеером
            self.reveal_btn = T.button(acts, REVEAL_TEXT, "folder-open", kind="ghost",
                                       height=28, size=13, command=lambda: _reveal(path))
            self.reveal_btn.pack(side="left")
        else:
            T.chip(acts, _("не записалась"), "bad").pack()
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
        T.reset_icons()
        self.TkdndVersion = tkinterdnd2.TkinterDnD._require(self)
        self.cfg = load_config()
        from core import audio_out
        audio_out.set_output(self.cfg.get("audio_out"))
        self.title(APP_NAME)
        # окно растёт вместе с масштабом текста (120 % по умолчанию), но не больше экрана
        k = float(self.cfg.get("scale", DEFAULT_SCALE))
        sw, sh = self.winfo_screenwidth() - 40, self.winfo_screenheight() - 90
        self.geometry(f"{min(sw, int(1180 * k))}x{min(sh, int(780 * k))}")
        self.minsize(min(sw, int(1040 * k)), min(sh, int(700 * k)))
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
        self._prog_hist: list = []
        self._wait_lock = False
        self._scale = float(self.cfg.get("scale", DEFAULT_SCALE))
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
        self.bind("<space>", self._on_space)          # «Прочитать вслух»: ▶ / ❚❚ (3.8)
        # размер текста с клавиатуры (Единый стандарт 1.5): Cmd+= / Cmd+− / Cmd+0
        mod = "Command" if sys.platform == "darwin" else "Control"
        for seq, d in (("equal", 0.1), ("plus", 0.1), ("minus", -0.1), ("0", None)):
            self.bind_all(f"<{mod}-{seq}>", lambda e, d=d: self._scale_by(d))
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        ctk.CTkFrame(self, width=1, fg_color=T.BORDER, corner_radius=0).grid(
            row=0, column=1, sticky="ns")
        self.content = ctk.CTkFrame(self, fg_color=T.BG, corner_radius=0)
        self.content.grid(row=0, column=2, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(1, weight=1)
        self._build_topbar()
        # страницы строим заранее: построение в момент щелчка по меню давало мигание
        self.pages = {"book": self._build_book_page(),
                      "say": self._build_say_page(),
                      "voices": self._build_voices_page(),
                      "help": self._build_help_page(),
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

        def purge_trash():                # корзина: старше 30 дней - навсегда (задание 12)
            try:
                from core import trash
                trash.purge_old()
            except Exception as e:
                log_problem("корзина не почистилась", e, show=False)
        threading.Thread(target=purge_trash, daemon=True).start()

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
        edit.add_command(label=_("Скопировать"), accelerator="Cmd+C",
                         command=lambda: self._menu_event("<<Copy>>"))
        edit.add_command(label=_("Выделить всё"), accelerator="Cmd+A",
                         command=lambda: self._menu_event("<<SelectAll>>"))
        edit.add_separator()
        edit.add_command(label=_("Скопировать подробности"), command=lambda: self.copy_log())
        menubar.add_cascade(label=_("Правка"), menu=edit)
        self.configure(menu=menubar)

    def _menu_event(self, event: str):
        w = self.focus_get()
        if w is not None:
            w.event_generate(event)

    def _build_sidebar(self):
        collapsed = bool(self.cfg.get("menu_collapsed", False))
        side = ctk.CTkFrame(self, width=64 if collapsed else 220, fg_color=T.SURFACE,
                            corner_radius=0)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        side.pack_propagate(False)
        self.side = side
        head = ctk.CTkFrame(side, fg_color="transparent")
        head.pack(fill="x", padx=(22 if not collapsed else 10, 10), pady=(18, 2))
        if not collapsed:
            T.label(head, APP_NAME, 19, bold=True, color=T.ACCENT, anchor="w").pack(side="left")
        # [≡] - свернуть меню до значков (стандарт интерфейса, п. 1; запоминается)
        self.menu_btn = ctk.CTkButton(head, text="≡", width=34, height=30, corner_radius=8,
                                      fg_color="transparent", hover_color=T.SURFACE2,
                                      text_color=T.MUTED, font=T.font(18),
                                      command=self._toggle_menu)
        self.menu_btn.pack(side="right")
        if not collapsed:
            T.label(side, _(APP_TAGLINE), 12, color=T.MUTED, anchor="w").pack(
                fill="x", padx=22, pady=(0, 18))
        else:
            ctk.CTkFrame(side, height=18, fg_color="transparent").pack()
        self.nav = {}
        for key, text, ic in (("book", _("Озвучка"), "audio-lines"), ("say", _("Прочитать вслух"), "play"),
                              ("voices", _("Голоса"), "mic")):
            self.nav[key] = self._nav_btn(side, "" if collapsed else text, ic, key)
            self.nav[key].pack(fill="x", padx=12, pady=1)
        bottom = ctk.CTkFrame(side, fg_color="transparent")
        bottom.pack(side="bottom", fill="x", padx=12, pady=(0, 14))
        ctk.CTkFrame(bottom, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=(0, 8))
        # внизу меню - всегда «Справка», «Настройки», «Тема», версия (стандарт, п. 1)
        self.nav["help"] = self._nav_btn(bottom, "" if collapsed else _("Справка"),
                                         "circle-alert", "help")
        self.nav["help"].pack(fill="x", pady=1)
        self.nav["settings"] = self._nav_btn(bottom, "" if collapsed else _("Настройки"),
                                             "settings", "settings")
        self.nav["settings"].pack(fill="x", pady=1)
        self.theme_btn = self._nav_btn(bottom, "", "sun-moon", None)
        self.theme_btn.configure(command=self._cycle_theme)
        self.theme_btn.pack(fill="x", pady=1)
        self._theme_label()
        self.version_lbl = T.label(bottom, f"{APP_NAME} {VERSION}" if collapsed else
                                   _('{0} {1}\nлокально {2}').format(APP_NAME, VERSION,
                                                                     tr_msg(osutil.HERE)),
                                   11, color=T.FAINT, anchor="w", justify="left")
        self.version_lbl.pack(fill="x", padx=10, pady=(8, 0))

    # ================= каркас по стандарту интерфейса (3.7) =================
    def _update_crumb(self):
        """Путь в верхней строке: «Раздел › Объект» (стандарт интерфейса, п. 1)."""
        if not hasattr(self, "crumb"):
            return
        t = self._page_title(getattr(self, "page", "book"))
        if self.page == "book" and getattr(self, "book_path", None):
            t += " › " + _short(self.book_title.cget("text"), 60)
        self.crumb.configure(text=t)

    def _page_title(self, key: str) -> str:
        return {"book": _("Озвучка"), "say": _("Прочитать вслух"), "voices": _("Голоса"),
                "help": _("Справка"), "settings": _("Настройки")}.get(key, "")

    def _build_topbar(self):
        """Верхняя строка 48 px: раздел, состояние, RU | EN, «Aa» (стандарт, п. 1)."""
        bar = ctk.CTkFrame(self.content, height=48, fg_color=T.BG, corner_radius=0)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)
        self.crumb = T.label(bar, "", 14, bold=True, anchor="w")
        self.crumb.pack(side="left", padx=24)
        self.aa_btn = ctk.CTkButton(bar, text="Aa", width=44, height=30, corner_radius=8,
                                    fg_color="transparent", hover_color=T.SURFACE2,
                                    text_color=T.TEXT, font=T.font(14, bold=True),
                                    command=self._aa_popup)
        self.aa_btn.pack(side="right", padx=(6, 24))
        self.lang_seg = ctk.CTkSegmentedButton(bar, values=["RU", "EN"], height=30, width=90,
                                               font=T.font(13), selected_color=T.ACCENT,
                                               selected_hover_color=T.ACCENT_HOVER,
                                               command=self._on_lang)
        self.lang_seg.set(i18n.lang().upper())
        self.lang_seg.pack(side="right")
        self.gpu_lbl = T.label(bar, "", 12, color=T.WARN, anchor="e")
        self.gpu_lbl.pack(side="right", padx=12)
        ctk.CTkFrame(self.content, height=1, fg_color=T.BORDER, corner_radius=0).grid(
            row=0, column=0, sticky="sew")
        self.after(1500, self._gpu_status)

    def _gpu_status(self):
        """Состояние справа: видеокарта занята другой программой (общий замок)."""
        try:
            from core import busy
            owner = busy.lock_owner()
            self.gpu_lbl.configure(text=_('Видеокарта занята: «{0}»').format(owner) if owner else "")
        except Exception:
            pass                                  # состояние - только подсказка
        if self.winfo_exists():
            self.after(10000, self._gpu_status)

    def _aa_popup(self):
        """«Aa» (задание 12, Единый стандарт 1.5): панель под кнопкой В САМОМ окне, а не
        новое окно - повторное нажатие закрывает, второй панели не бывает, в
        полноэкранном режиме ничего не уезжает на другой экран. 80-200 %, шаг 10 %."""
        pop = getattr(self, "_aa_pop", None)
        if pop is not None and pop.winfo_exists() and pop.winfo_ismapped():
            self._aa_close()
            return
        if pop is None or not pop.winfo_exists():
            pop = ctk.CTkFrame(self, fg_color=T.SURFACE, border_width=1,
                               border_color=T.BORDER_STRONG, corner_radius=10)
            T.label(pop, _("Размер текста"), 12, color=T.MUTED, anchor="w").pack(
                fill="x", padx=14, pady=(10, 2))
            row = ctk.CTkFrame(pop, fg_color="transparent")
            row.pack(padx=10, pady=(0, 4))
            T.button(row, "A−", None, kind="ghost", height=30, width=44,
                     command=lambda: self._scale_by(-0.1)).pack(side="left")
            self._aa_lbl = T.label(row, "", 14, width=70)
            self._aa_lbl.pack(side="left")
            T.button(row, "A+", None, kind="ghost", height=30, width=44,
                     command=lambda: self._scale_by(0.1)).pack(side="left")
            T.button(pop, _("Сбросить"), "refresh-cw", kind="ghost", height=28,
                     command=lambda: self._scale_by(None)).pack(pady=(0, 10))
            self._aa_pop = pop
            if not getattr(self, "_aa_bound", False):
                self.bind_all("<Button-1>", self._aa_outside, add="+")
                self.bind("<Escape>", lambda e: self._aa_close(), add="+")
                self._aa_bound = True
        self._aa_show()
        pop.place(relx=1.0, rely=0.0, x=-18, y=46, anchor="ne")
        pop.lift()
        self._aa_win = pop                      # для проверок: одна панель на окно

    def _aa_show(self):
        lbl = getattr(self, "_aa_lbl", None)
        if lbl is not None and lbl.winfo_exists():
            lbl.configure(text=f"{round(self._scale * 100)} %")

    def _aa_close(self):
        pop = getattr(self, "_aa_pop", None)
        if pop is not None and pop.winfo_exists():
            pop.place_forget()

    def _aa_outside(self, event):
        """Щелчок мимо панели (и мимо «Aa») закрывает её."""
        pop = getattr(self, "_aa_pop", None)
        if pop is None or not pop.winfo_exists() or not pop.winfo_ismapped():
            return
        w = str(event.widget)
        if w.startswith(str(pop)) or w.startswith(str(self.aa_btn)):
            return
        self._aa_close()

    def _scale_by(self, d: float | None):
        """A− / A+ / «Сбросить» и Cmd+= / Cmd+− / Cmd+0."""
        self._set_scale(DEFAULT_SCALE if d is None else self._scale + d)
        self._aa_show()
        return "break"

    def _set_scale(self, f: float):
        self._scale = max(SCALE_MIN, min(SCALE_MAX, round(f, 2)))
        ctk.set_widget_scaling(self._scale)
        self.cfg["scale"] = self._scale
        save_config(self.cfg)

    def _toggle_menu(self):
        self.cfg["menu_collapsed"] = not bool(self.cfg.get("menu_collapsed", False))
        save_config(self.cfg)
        self._restart()

    def _on_lang(self, v: str):
        lang = v.lower()
        if lang == i18n.lang():
            return
        if self.running:
            messagebox.showinfo(_("Язык"), _("Язык можно сменить, когда озвучка закончится."))
            self.lang_seg.set(i18n.lang().upper())
            return
        self.cfg["ui_lang"] = lang
        save_config(self.cfg)
        i18n.set_lang(lang)
        self._restart()

    def _restart(self):
        """Пересобрать окно (язык, меню): книга и раздел остаются выбранными."""
        global _RESTART
        if self.running:
            return
        self._live_stop()
        _RESTART = {"book": self.book_path, "page": self.page}
        if self._srv is not None:
            self._srv.stop()
        self.quit()

    def _build_help_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        T.label(page, _("Как пользоваться"), 22, bold=True, anchor="w").grid(
            row=0, column=0, sticky="ew", padx=24, pady=(22, 10))
        box = ctk.CTkTextbox(page, wrap="word", font=T.font(14), border_width=0,
                             fg_color=T.SURFACE)
        box.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 20))
        box.insert("1.0", i18n.text("help").format(v=VERSION))
        box.configure(state="disabled")
        self._enable_copy(box)
        return page

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
                p.grid(row=1, column=0, sticky="nsew")
            else:
                p.grid_remove()
        self.page = key
        self._update_crumb()
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
        names = {"system": _("Системная"), "light": _("Светлая"), "dark": _("Тёмная")}
        self.theme_btn.configure(text=_('Тема: {0}').format(names.get(self.cfg.get('theme', 'system'), 'Системная')))

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
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(22, 14))
        T.label(head, _("Озвучка"), 22, bold=True, anchor="w").pack(fill="x")
        T.label(head, _('Книга - в аудиокнигу выбранным голосом; всё считается {0}.').format(_(osutil.HERE)),
                13, color=T.MUTED, anchor="w", justify="left", wraplength=820).pack(fill="x", pady=(2, 0))

        left = ctk.CTkFrame(page, fg_color="transparent")
        left.grid(row=1, column=0, sticky="nsew", padx=(24, 8))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.book_card = T.card(left)
        self.book_card.grid(row=0, column=0, sticky="nsew")
        self._build_empty(self.book_card)
        self._build_book_info(self.book_card)
        self._build_details(left)

        opts = T.card(page)
        opts.grid(row=1, column=1, sticky="nsew", padx=(8, 24))
        self._build_options(opts)

        bar = ctk.CTkFrame(page, fg_color="transparent")
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=24, pady=(14, 20))
        lft = ctk.CTkFrame(bar, fg_color="transparent")
        lft.pack(side="left", fill="x", expand=True, padx=(0, 18))
        prow = ctk.CTkFrame(lft, fg_color="transparent")
        prow.pack(fill="x")
        self.progress = ctk.CTkProgressBar(prow, height=6)
        self.progress.set(0)
        self.progress.pack(side="left", fill="x", expand=True, pady=(4, 0))
        self.timer = T.label(prow, _fmt_clock(0), 13, color=T.MUTED, width=130, anchor="e")
        self.timer.pack(side="right", padx=(10, 0))
        self.status = T.label(lft, _("Перетащите книгу - и нажмите «Озвучить книгу»."), 13,
                              color=T.MUTED, anchor="w")
        self.status.pack(fill="x", pady=(4, 0))
        # видеокарта занята другой программой (3.7) - строка с кнопкой, по надобности
        self.busy_line = ctk.CTkFrame(lft, fg_color="transparent")
        self.busy_note = T.label(self.busy_line, "", 12, color=T.WARN, anchor="w")
        self.busy_note.pack(side="left")
        self.busy_btn = T.button(self.busy_line, _("Подождать, пока освободится"), "pause",
                                 kind="ghost", height=26, size=12, width=230,
                                 command=self._wait_free)
        self.busy_btn.pack(side="left", padx=(10, 0))
        prow2 = ctk.CTkFrame(lft, fg_color="transparent")
        prow2.pack(fill="x", pady=(4, 0))
        self.part_label = T.label(prow2, "", 12, color=T.FAINT, anchor="w", width=230)
        self.part_label.pack(side="left")
        self.part_progress = ctk.CTkProgressBar(prow2, height=4, progress_color=T.FAINT)
        self.part_progress.set(0)
        self.part_progress.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.restart_btn = T.button(bar, _("Начать заново"), "refresh-cw", kind="ghost", height=42,
                                    size=13, width=150, command=self.do_restart)
        self.restart_btn.configure(border_width=1, border_color=T.BORDER_STRONG)
        # «Пауза» видна только во время озвучки и пробника (задание 12: неактивная без
        # объяснения кнопка - «мёртвый конец»)
        self.pause_btn = T.button(bar, _("Пауза"), "pause", height=42, size=14, width=130,
                                  command=self.do_pause)
        self.start_btn = T.button(bar, _("Озвучить книгу"), "play", kind="primary", width=230,
                                  height=42, size=15, command=self.on_main_button)
        self.start_btn.pack(side="right")
        self._start_style()
        return page

    def _start_style(self):
        """Одно главное действие на экране: без книги главная - «Выбрать книгу», а
        «Озвучить книгу» неактивна (строка состояния говорит, что сделать)."""
        if self.running:
            return
        if self.book_path:
            self.start_btn.configure(state="normal")
            T.restyle(self.start_btn, "primary", "play")
        else:
            T.restyle(self.start_btn, "default", "play")
            self.start_btn.configure(state="disabled")

    def _pause_visible(self, on: bool):
        if on and not self.pause_btn.winfo_manager():
            self.pause_btn.pack(side="right", padx=(8, 0), before=self.start_btn)
        elif not on:
            self.pause_btn.pack_forget()

    def _build_empty(self, parent):
        self.empty = ctk.CTkFrame(parent, fg_color=T.SURFACE, border_width=2,
                                  border_color=T.BORDER_STRONG, corner_radius=T.RADIUS)
        self.empty.pack(fill="both", expand=True, padx=14, pady=14)
        inner = ctk.CTkFrame(self.empty, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ic = ctk.CTkLabel(inner, text="", image=T.icon("upload", T.ACCENT, 36))
        ic.pack(pady=(0, 10))
        title = T.label(inner, _("Перетащите книгу сюда"), 17, bold=True)
        title.pack()
        sub = T.label(inner, "txt · md · docx · epub · fb2 · pdf", 13, color=T.MUTED,
                      justify="center")
        sub.pack(pady=(4, 14))
        T.button(inner, _("Выбрать книгу"), "plus", kind="primary", width=170,
                 command=self.choose_book).pack()
        for w in (self.empty, inner, ic, title, sub):
            w.bind("<Button-1>", lambda e: self.choose_book())
        # «Недавние книги» (задание 12): до 5, щелчок открывает
        self.recent_box = ctk.CTkFrame(inner, fg_color="transparent")
        self.recent_box.pack(fill="x", pady=(22, 0))
        self._recent_show()

    def _recent_list(self) -> list[str]:
        return [p for p in (self.cfg.get("recent_books") or []) if os.path.isfile(p)][:RECENT_MAX]

    def _recent_add(self, path: str):
        p = os.path.abspath(path)
        lst = [x for x in (self.cfg.get("recent_books") or []) if x != p]
        self.cfg["recent_books"] = [p] + lst[:RECENT_MAX - 1]
        save_config(self.cfg)
        self._recent_show()

    def _recent_show(self):
        box = getattr(self, "recent_box", None)
        if box is None:
            return
        for w in box.winfo_children():
            w.destroy()
        lst = self._recent_list()
        if not lst:
            return
        T.label(box, _("Недавние книги"), 13, bold=True, color=T.MUTED, anchor="w").pack(fill="x")
        for p in lst:
            name = os.path.splitext(os.path.basename(p))[0]
            b = T.button(box, _short(name, 46), "file-text", kind="ghost", height=26, size=12,
                         command=lambda p=p: self._open_recent(p))
            b.configure(anchor="w")
            b.pack(fill="x", pady=(2, 0))
            T.label(box, _short(os.path.dirname(p), 60), 11, color=T.FAINT, anchor="w").pack(
                fill="x", padx=(30, 0))

    def _open_recent(self, p: str):
        if self.running:
            return self._busy_book_msg()
        if os.path.isfile(p):
            self._set_book(p)
        else:
            self._recent_show()

    def _busy_book_msg(self):
        messagebox.showinfo(_("Идёт озвучка"), _("Идёт озвучка - другую книгу можно открыть "
                                                "после «Остановить»."))

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
        # служебные файлы книги (задание 12): размер, «Открыть», «Очистить»
        self.svc_line = ctk.CTkFrame(top, fg_color="transparent")
        self.svc_lbl = T.label(self.svc_line, "", 12, color=T.MUTED, anchor="w")
        self.svc_lbl.pack(side="left")
        self.svc_open = T.button(self.svc_line, _("Открыть"), "folder-open", kind="ghost", height=24,
                                 size=12, width=96, command=self._service_open)
        self.svc_open.pack(side="left", padx=(8, 0))
        self.svc_clear = T.button(self.svc_line, _("Очистить"), "x", kind="ghost", height=24,
                                  size=12, width=100, command=self._service_clear)
        self.svc_clear.pack(side="left", padx=(4, 0))
        ctk.CTkFrame(self.info, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=(8, 0))
        hdr = ctk.CTkFrame(self.info, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(10, 6))
        self.parts_title = T.label(hdr, _("Готовые части"), 15, bold=True, anchor="w")
        self.parts_title.pack(side="left")
        self.rows = ctk.CTkScrollableFrame(self.info, fg_color="transparent", corner_radius=0)
        self.rows.pack(fill="both", expand=True, padx=1, pady=(0, 6))
        self.no_parts = T.label(self.rows, _("Пока ни одной. Части появятся здесь по мере "
                                           "готовности - их можно слушать сразу."),
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
        self.details_btn = T.button(line, _("Подробности"), "file-text", kind="ghost", height=28,
                                    size=13, width=140, command=self.toggle_details)
        self.details_btn.pack(side="left")
        self.copy_btn = T.button(line, _("Копировать"), "copy", kind="ghost", height=28, size=13,
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
        # без прокрутки (задание 12): при 1440×900 и «Aa» 100-120 % панель целиком на
        # экране - подписи и меню в одну строку, редкое - ссылками
        o = ctk.CTkFrame(card, fg_color="transparent", corner_radius=0, width=300)
        o.pack(fill="both", expand=True, padx=18, pady=12)
        self.opts_box = o                                   # для проверок
        ctk.CTkFrame(o, width=290, height=0, fg_color="transparent").pack()
        T.section(o, _("Голос")).pack(fill="x")
        self.voice_var = ctk.StringVar(value="")
        self.voice_menu = ctk.CTkOptionMenu(o, variable=self.voice_var, values=[""], height=32,
                                            font=T.font(13), dropdown_font=T.font(13),
                                            command=lambda *_a: self._on_voice())
        self.voice_menu.pack(fill="x", pady=(6, 2))
        self.voice_hint = T.label(o, "", 12, color=T.MUTED, anchor="w", justify="left",
                                  wraplength=290)
        self.voice_hint.pack(fill="x")
        # звучание - одно место, страница «Голоса» (3.2.4а)
        link = T.button(o, _("Звучание голоса ›"), None, kind="ghost", height=24, size=12, width=10,
                        command=lambda: self._open_voice_fx(self._voice_obj()))
        link.configure(anchor="w")
        link.pack(anchor="w")

        def row(title):
            line = ctk.CTkFrame(o, fg_color="transparent")
            line.pack(fill="x", pady=(8, 0))
            T.label(line, title, 13, anchor="w", width=104).pack(side="left")
            return line
        # показывается перевод, хранится исходное значение (_len / _fmt)
        self.len_var = ctk.StringVar(value=_(self.cfg.get("len", "15 минут")
                                             if self.cfg.get("len") in FILE_LENGTHS else "15 минут"))
        self.len_menu = ctk.CTkOptionMenu(row(_("Длина частей")), variable=self.len_var,
                                          values=[_(x) for x in FILE_LENGTHS], height=30,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_a: (self._remember(), self.reestimate()))
        self.len_menu.pack(side="left", fill="x", expand=True)
        fmt0 = self.cfg.get("fmt") if self.cfg.get("fmt") in FORMATS else DEFAULT_FORMAT
        self.fmt_var = ctk.StringVar(value=self._fmt_label(fmt0))
        self.fmt_menu = ctk.CTkOptionMenu(row(_("Формат")), variable=self.fmt_var,
                                          values=[self._fmt_label(x) for x in FORMATS], height=30,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_a: (self._remember(), self.show_estimate()))
        self.fmt_menu.pack(side="left", fill="x", expand=True)
        T.label(o, _("Время озвучки от формата почти не зависит: кодирование - доли процента "
                     "от синтеза."), 11, color=T.FAINT, anchor="w", justify="left",
                wraplength=290).pack(fill="x", pady=(4, 0))

        ctk.CTkFrame(o, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=10)
        line = ctk.CTkFrame(o, fg_color="transparent")
        line.pack(fill="x")
        T.label(line, _("Движок"), 13, anchor="w", width=104).pack(side="left")
        self.profile_chip = T.chip(line, "", "acc")
        self.profile_chip.pack(side="left")
        self.accel_hint = T.label(o, "", 12, color=T.MUTED, anchor="w", justify="left",
                                  wraplength=290)
        self.accel_hint.pack(fill="x", pady=(2, 0))

        ctk.CTkFrame(o, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x", pady=10)
        self.preview_btn = T.button(o, _("Проверить голос (1 мин)"), "play", height=34, size=14,
                                    command=self.do_preview)
        self.preview_btn.pack(fill="x")
        T.label(o, _("Минута из середины самой длинной части книги - послушать голос до "
                     "многочасовой озвучки."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=290).pack(fill="x", pady=(4, 0))
        self.readtext_btn = T.button(o, _("Как будет прочитано"), "file-text", kind="ghost",
                                     height=30, size=13, command=self.show_read_text)
        self.readtext_btn.pack(fill="x", pady=(6, 0))
        self._setting_widgets = [self.voice_menu, self.len_menu, self.fmt_menu]

    @staticmethod
    def _fmt_label(key: str) -> str:
        """«mp3 192 кбит/с (обычный) · ~82 МБ/ч» - формат и размер часа звука."""
        mb = FORMAT_KBPS.get(key, 192) * 3600 / 8 / 1024
        return f"{_(key)} · ~{mb:.0f} " + _("МБ/ч")

    # ================= страница «Голоса» =================
    def _build_voices_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent", corner_radius=0)
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        head = ctk.CTkFrame(page, fg_color="transparent")
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(22, 14))
        T.label(head, _("Голоса"), 22, bold=True, anchor="w").pack(fill="x")
        T.label(head, _("Клон - из любой записи диктора; готовые голоса работают сразу."),
                13, color=T.MUTED, anchor="w", justify="left", wraplength=820).pack(fill="x", pady=(2, 0))

        lst = T.card(page)
        lst.grid(row=1, column=0, sticky="nsew", padx=(24, 8), pady=(0, 20))
        hdr = ctk.CTkFrame(lst, fg_color="transparent")
        hdr.pack(fill="x", padx=14, pady=(10, 8))
        T.label(hdr, _("Все голоса"), 15, bold=True, anchor="w").pack(side="left")
        self.show_hidden_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(hdr, text=_("Показать скрытые"), variable=self.show_hidden_var,
                        font=T.font(13), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                        border_color=T.BORDER_STRONG, text_color=T.TEXT, checkbox_width=18,
                        checkbox_height=18, command=self._refresh_voice_rows).pack(side="left", padx=(14, 0))
        self.add_voice_btn = T.button(hdr, _("Добавить образец…"), "plus", kind="primary", height=30,
                                      size=13, width=170, command=self.load_sample)
        self.add_voice_btn.pack(side="right")
        T.button(hdr, "", "folder-open", kind="ghost", height=30, width=34,          # папка голосов
                 command=lambda: _open_path(default_voices_dir())).pack(side="right", padx=6)
        ctk.CTkFrame(lst, height=1, fg_color=T.BORDER, corner_radius=0).pack(fill="x")
        # загрузка голоса: полоса и строка хода - внизу списка (3.8: всё о голосах здесь)
        vm_foot = ctk.CTkFrame(lst, fg_color="transparent")
        vm_foot.pack(side="bottom", fill="x", padx=14, pady=(0, 8))
        self.vm_prog = ctk.CTkProgressBar(vm_foot, height=6)
        self.vm_prog.set(0)
        self.vm_note = T.label(vm_foot, "", 12, color=T.MUTED, anchor="w", justify="left",
                               wraplength=520)
        self.vm_note.pack(fill="x")
        self._vm_busy = None
        self.voice_rows = ctk.CTkScrollableFrame(lst, fg_color="transparent", corner_radius=0)
        self.voice_rows.pack(fill="both", expand=True, padx=1, pady=(0, 6))

        fx = T.card(page)
        fx.grid(row=1, column=1, sticky="nsew", padx=(8, 24), pady=(0, 20))
        self.fx_box = ctk.CTkFrame(fx, fg_color="transparent")
        self.fx_box.pack(fill="both", expand=True, padx=18, pady=16)
        ctk.CTkFrame(self.fx_box, width=300, height=0, fg_color="transparent").pack()
        T.section(self.fx_box, _("Звучание голоса")).pack(fill="x")
        self.fx_title = T.label(self.fx_box, _("Выберите голос слева"), 16, bold=True, anchor="w")
        self.fx_title.pack(fill="x", pady=(6, 0))
        self.fx_src = T.label(self.fx_box, "", 12, color=T.MUTED, anchor="w", justify="left",
                              wraplength=300)
        self.fx_src.pack(fill="x", pady=(0, 6))
        T.label(self.fx_box, _("Поправки запоминаются для голоса и применяются при каждой "
                             "озвучке. Интонацию так не поменять - она приходит из образца."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=300).pack(fill="x", pady=(0, 10))
        for title, key, lo, hi, unit in ((_("Темп"), "speed", 0.8, 1.25, "×"),
                                         (_("Высота"), "pitch", -4, 4, _(" полутона")),
                                         (_("Бас"), "bass", -6, 6, _(" дБ")),
                                         (_("Яркость"), "brightness", -6, 6, _(" дБ"))):
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
        self.fx_play_btn = T.button(btns, _("Прослушать"), "play", kind="default", height=34, width=140,
                                    command=self._fx_play)
        self.fx_play_btn.pack(side="left")
        T.button(btns, _("Сбросить"), "refresh-cw", kind="ghost", height=34, width=110,
                 command=self._fx_reset).pack(side="left", padx=6)
        self.fx_save_btn = T.button(btns, _("Сохранить"), "save", height=34, width=120,
                                    command=self._fx_save)
        self.fx_save_btn.pack(side="right")
        self.fx_note = T.label(self.fx_box, "", 12, color=T.OK, anchor="w")
        self.fx_note.pack(fill="x", pady=(4, 0))
        # «Дополнительно» (задание 12, п. 8): свои веса, выборка Qwen, шумоподавление
        self.fx_more = ctk.CTkFrame(self.fx_box, fg_color="transparent")
        self.fx_more.pack(fill="x", pady=(4, 0))
        T.section(self.fx_more, _("Дополнительно")).pack(fill="x")
        self.fx_info = T.label(self.fx_more, "", 12, color=T.MUTED, anchor="w", justify="left",
                               wraplength=300)
        self.fx_info.pack(fill="x", pady=(4, 0))
        self.dn_box = ctk.CTkFrame(self.fx_more, fg_color="transparent")
        T.label(self.dn_box, _("Шумоподавление (убирает шипение записи)"), 13, anchor="w").pack(fill="x", pady=(6, 2))
        self.dn_var = ctk.StringVar(value=_("нет"))
        self.dn_seg = ctk.CTkSegmentedButton(
            self.dn_box, values=[_("нет"), _("мягкое"), _("сильное")], variable=self.dn_var,
            height=28, font=T.font(12), selected_color=T.ACCENT, selected_hover_color=T.ACCENT_HOVER,
            command=lambda v: self._fx_denoise(v))
        self.dn_seg.pack(fill="x")
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
        head.pack(fill="x", padx=24, pady=(22, 14))
        T.label(head, _("Настройки"), 22, bold=True, anchor="w").pack(fill="x")

        c = T.card(page)
        c.pack(fill="x", padx=24, pady=(0, 14))
        b = ctk.CTkFrame(c, fg_color="transparent")
        b.pack(fill="x", padx=18, pady=16)
        T.section(b, _("Движок")).pack(fill="x")
        T.label(b, _("Выбран слепым прослушиванием главы: лучший - «Основной». Готовые "
                   "голоса Silero включаются выбором голоса."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", pady=(2, 8))
        self.profile_var = ctk.StringVar(value=profiles.get(self.cfg.get("profile")).key)
        self._profile_radios = []
        for p in [x for x in profiles.PROFILES.values() if profiles.available(x)]:
            line = ctk.CTkFrame(b, fg_color="transparent")
            line.pack(fill="x", pady=(8, 0))
            r = ctk.CTkRadioButton(line, text=_(p.title), value=p.key, variable=self.profile_var,
                                   font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                                   border_color=T.BORDER_STRONG, text_color=T.TEXT,
                                   command=self._on_profile)
            r.pack(side="left")
            # «Послушать»: та же фраза, что в «Голосах», выбранным голосом и этим движком
            pb = T.button(line, _("Послушать"), "play", kind="ghost", height=26, size=12,
                          width=110, command=lambda k=p.key: self._profile_listen(k))
            pb.pack(side="left", padx=(10, 0))
            self._prof_btns[p.key] = pb
            self._profile_radios.append(r)
            T.label(b, _(p.hint), 12, color=T.MUTED, anchor="w", justify="left",
                    wraplength=640).pack(fill="x", padx=(30, 0))

        cp = T.card(page)
        cp.pack(fill="x", padx=24, pady=(0, 14))
        bp = ctk.CTkFrame(cp, fg_color="transparent")
        bp.pack(fill="x", padx=18, pady=16)
        T.section(bp, _("Паузы")).pack(fill="x")
        T.label(bp, _("Тишина после предложения, после абзаца и между репликами диалога "
                    "(смена говорящего - поэтому длиннее)."), 12, color=T.MUTED, anchor="w",
                justify="left", wraplength=640).pack(fill="x", pady=(2, 6))
        saved = self.cfg.get("pauses", {})
        for key, title, dflt, lo, hi in PAUSES:
            line = ctk.CTkFrame(bp, fg_color="transparent")
            line.pack(fill="x", pady=3)
            T.label(line, _(title), 13, anchor="w", width=170).pack(side="left")
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
        T.button(bp, _("Как было"), "refresh-cw", kind="ghost", height=28, size=13, width=120,
                 command=self._pauses_reset).pack(anchor="w", pady=(6, 0))

        self.prof_note = T.label(b, "", 12, color=T.MUTED, anchor="w")
        self.prof_note.pack(fill="x", pady=(8, 0))

        cq = T.card(page)
        cq.pack(fill="x", padx=24, pady=(0, 14))
        bq = ctk.CTkFrame(cq, fg_color="transparent")
        bq.pack(fill="x", padx=18, pady=16)
        T.section(bq, _("Качество / скорость")).pack(fill="x")
        T.label(bq, _("Для клона (основной и ESpeech). Модель и 12 шагов те же; меняется, на "
                    "скольких шагах модель сверяется с текстом дважды."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", pady=(2, 6))
        qk = self.cfg.get("quality")
        self.quality_var = ctk.StringVar(value=qk if qk in profiles.QUALITY else profiles.DEFAULT_QUALITY)
        self._quality_radios = []
        for q in profiles.QUALITY.values():
            r = ctk.CTkRadioButton(bq, text=_(q.title), value=q.key, variable=self.quality_var,
                                   font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                                   border_color=T.BORDER_STRONG, text_color=T.TEXT,
                                   command=self._on_quality)
            r.pack(anchor="w", pady=(8, 0))
            self._quality_radios.append(r)
            T.label(bq, _(q.hint), 12, color=T.MUTED, anchor="w", justify="left",
                    wraplength=640).pack(fill="x", padx=(30, 0))

        cs = T.card(page)
        cs.pack(fill="x", padx=24, pady=(0, 14))
        bs = ctk.CTkFrame(cs, fg_color="transparent")
        bs.pack(fill="x", padx=18, pady=16)
        T.section(bs, _("Проверка")).pack(fill="x")
        self.sound_var = ctk.BooleanVar(value=bool(self.cfg.get("sound_check", True)))
        self.sound_chk = ctk.CTkCheckBox(
            bs, text=_("Проверка слов по звуку"), variable=self.sound_var,
            font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
            border_color=T.BORDER_STRONG, text_color=T.TEXT, command=self._on_strict)
        self.sound_chk.pack(anchor="w", pady=(8, 0))
        T.label(bs, _("Текст книги укладывается на звук моделью без языковой догадки: "
                    "съеденное, смазанное или заменённое слово видно "
                    "по звуку, и кусок пересчитывается с другим зерном (до двух раз); "
                    "берётся вариант с лучшим худшим словом. Края кусков проверяются всегда."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", padx=(30, 0))
        # строгая проверка 3.2 (текст Whisper против текста книги) убрана из окна
        # в 3.3: на слух выигрыша не дала, а время +50 %; проверка по звуку лучше
        # на размеченном наборе (9 из 10 мест против 7 из 10)
        self.strict_var = ctk.BooleanVar(value=False)

        # служебные файлы книги и корзина (задание 12, Единый стандарт 3.2)
        cc = T.card(page)
        cc.pack(fill="x", padx=24, pady=(0, 14))
        bc = ctk.CTkFrame(cc, fg_color="transparent")
        bc.pack(fill="x", padx=18, pady=16)
        T.section(bc, _("Служебные файлы книги")).pack(fill="x")
        self.cleanup_var = ctk.BooleanVar(value=bool(self.cfg.get("cleanup_service", True)))
        self.cleanup_chk = ctk.CTkCheckBox(
            bc, text=_("Удалять служебные файлы после готовой книги"), variable=self.cleanup_var,
            font=T.font(14), fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
            border_color=T.BORDER_STRONG, text_color=T.TEXT, command=self._on_cleanup)
        self.cleanup_chk.pack(anchor="w", pady=(8, 0))
        T.label(bc, _("Текст с ударениями, куски, сырые части. Пока книга не готова, они нужны "
                    "для «Продолжить» и пересчёта кусков; готовая книга - в корзину на 30 дней."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(fill="x", padx=(30, 0))
        self._build_trash_card(page)

        # куда идёт звук потоком (3.8: «звук идёт не в наушники»)
        cao = T.card(page)
        cao.pack(fill="x", padx=24, pady=(0, 14))
        bao = ctk.CTkFrame(cao, fg_color="transparent")
        bao.pack(fill="x", padx=18, pady=16)
        T.section(bao, _("Устройство вывода")).pack(fill="x")
        T.label(bao, _("«Как в системе» - звук идёт туда же, куда у системы, и сам переходит в "
                     "наушники, когда их подключают."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", pady=(2, 6))
        from core import audio_out
        try:
            devs = audio_out.outputs()
        except Exception:
            devs = []
        cur = self.cfg.get("audio_out") or audio_out.SYSTEM
        self.out_var = ctk.StringVar(value=_("Как в системе") if cur == audio_out.SYSTEM else cur)
        self.out_menu = ctk.CTkOptionMenu(bao, variable=self.out_var, height=32, width=320,
                                          values=[_("Как в системе")] + devs,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_a: self._on_audio_out())
        self.out_menu.pack(anchor="w")
        self.out_menu._herald_data = True              # имена устройств - данные системы

        cr = T.card(page)
        cr.pack(fill="x", padx=24, pady=(0, 14))
        br = ctk.CTkFrame(cr, fg_color="transparent")
        br.pack(fill="x", padx=18, pady=16)
        T.section(br, _("Прочитать вслух")).pack(fill="x")
        self.say_clones_var = ctk.BooleanVar(value=bool(self.cfg.get("say_clones", False)))
        ctk.CTkCheckBox(br, text=_("Показывать клон-голоса в «Прочитать вслух»"),
                        variable=self.say_clones_var, font=T.font(14), fg_color=T.ACCENT,
                        hover_color=T.ACCENT_HOVER, border_color=T.BORDER_STRONG,
                        text_color=T.TEXT, command=self._on_say_clones).pack(anchor="w", pady=(8, 0))
        T.label(br, _("Клон в этой вкладке начинает говорить только через 10-50 с; быстрые "
                    "голоса - сразу. Для книги клон - лучший выбор."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", padx=(30, 0))

        csv = T.card(page)
        csv.pack(fill="x", padx=24, pady=(0, 14))
        bv = ctk.CTkFrame(csv, fg_color="transparent")
        bv.pack(fill="x", padx=18, pady=16)
        T.section(bv, _("Озвучка для других программ")).pack(fill="x")
        from core.speak_server import DEFAULT_PORT, DEFAULT_IDLE_MIN
        self.srv_var = ctk.BooleanVar(value=bool(self.cfg.get("speak_server", False)))
        self.srv_chk = ctk.CTkCheckBox(
            bv, text=_("Включить сервис /speak"), variable=self.srv_var, font=T.font(14),
            fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER, border_color=T.BORDER_STRONG,
            text_color=T.TEXT, command=self._srv_toggle)
        self.srv_chk.pack(anchor="w", pady=(8, 0))
        T.label(bv, _("Другие программы на этом компьютере (Legal Knowledge Desk, помощник по "
                    "настольной игре) просят Herald прочитать текст: POST /speak - и звук идёт "
                    "по мере готовности. Только 127.0.0.1, без интернета. Как вызвать - в README."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(
            fill="x", padx=(30, 0))
        line = ctk.CTkFrame(bv, fg_color="transparent")
        line.pack(fill="x", padx=(30, 0), pady=(8, 0))
        T.label(line, _("Порт"), 13, anchor="w").pack(side="left")
        self.srv_port = ctk.CTkEntry(line, width=80, height=28, font=T.font(13))
        self.srv_port.insert(0, str(self.cfg.get("speak_port", DEFAULT_PORT)))
        self.srv_port.pack(side="left", padx=(8, 18))
        self.srv_port.bind("<Return>", lambda e: self._srv_apply())
        self.srv_port.bind("<FocusOut>", lambda e: self._srv_apply())
        T.label(line, _("Выгружать модели после простоя"), 13, anchor="w").pack(side="left")
        idle_vals = [_("5 минут"), _("10 минут"), _("30 минут"), _("60 минут")]
        cur = _('{0} минут').format(int(self.cfg.get('speak_idle', DEFAULT_IDLE_MIN)))
        self.srv_idle = ctk.CTkOptionMenu(line, values=idle_vals, width=110, height=28,
                                          font=T.font(13), dropdown_font=T.font(13),
                                          command=lambda *_a: self._srv_apply())
        self.srv_idle.set(cur if cur in idle_vals else _("10 минут"))
        self.srv_idle.pack(side="left", padx=(8, 0))
        self.srv_note = T.label(bv, "", 12, color=T.MUTED, anchor="w")
        self.srv_note.pack(fill="x", padx=(30, 0), pady=(6, 0))

        c2 = T.card(page)
        c2.pack(fill="x", padx=24, pady=(0, 14))
        b2 = ctk.CTkFrame(c2, fg_color="transparent")
        b2.pack(fill="x", padx=18, pady=16)
        T.section(b2, _("Для опытных")).pack(fill="x")
        self.tech_label = T.label(b2, "", 12, color=T.MUTED, anchor="w", justify="left",
                                  wraplength=680)
        self.tech_label.pack(fill="x", pady=(6, 8))
        row = ctk.CTkFrame(b2, fg_color="transparent")
        row.pack(fill="x")
        # Ускорение NVIDIA (Windows): видно, только если есть видеокарта, а
        # считается на процессоре - torch с CUDA докачивается отдельно (~2,5 ГБ)
        self.nv_box = ctk.CTkFrame(b2, fg_color="transparent")
        self.nv_note = T.label(self.nv_box, _("Найдена видеокарта NVIDIA. Чтобы озвучка шла на ней, "
                                            "нужна сборка torch с CUDA (около 2,5 ГБ, скачивается "
                                            "отдельно, чтобы не раздувать установщик)."),
                               12, color=T.MUTED, anchor="w", justify="left", wraplength=640)
        self.nv_note.pack(fill="x")
        self.nv_btn = T.button(self.nv_box, _("Скачать ускорение NVIDIA"), "upload", height=30,
                               size=13, width=240, command=self._nvidia_download)
        self.nv_btn.pack(anchor="w", pady=(6, 8))
        T.button(row, _("Открыть журнал"), "circle-alert", height=30, size=13, width=170,
                 command=lambda: _open_path(LOG_PATH) if os.path.exists(LOG_PATH)
                 else messagebox.showinfo(_("Журнал"), _("Журнал пока пуст - проблем не было."))
                 ).pack(side="left")
        self.showlog_var = ctk.BooleanVar(value=bool(self.cfg.get("show_log", False)))
        ctk.CTkCheckBox(row, text=_("Показывать технический журнал на главной"),
                        variable=self.showlog_var, font=T.font(13), fg_color=T.ACCENT,
                        hover_color=T.ACCENT_HOVER, border_color=T.BORDER_STRONG,
                        text_color=T.TEXT, command=self._on_showlog).pack(side="left", padx=(14, 0))

        c3 = T.card(page)
        c3.pack(fill="x", padx=24, pady=(0, 22))
        box = ctk.CTkTextbox(c3, height=320, wrap="word", font=T.font(12), border_width=0,
                             fg_color=T.SURFACE)
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", i18n.text("about").format(v=VERSION, tag=_(APP_TAGLINE)))
        box.configure(state="disabled")
        self._update_tech()
        return outer

    def _update_tech(self):
        p = self.profile()
        kw = ", ".join(f"{k}={v}" for k, v in p.kwargs.items()) or "-"
        got = self.cfg.get("rtf", {}).get(self._rtf_key())
        measured = (_('замерено при последней озвучке: {0:.2f}').format(got) if got
                    else _("замера ещё не было - оценка по стенду"))
        self.tech_label.configure(text=(
            _('«{0}»: движок {1}, параметры {2}. Счёт на секунду звука (RTF) ~{3:.2f} с на Apple M4, {4}. Шаги - число уточнений картинки звука (NFE), расписание EPSS - прореженные шаги из работы Fast F5-TTS. Готовые куски запоминаются (кэш синтеза), поэтому смена пауз или формата пересобирает книгу за минуты. Журнал: ').format(_(p.title), p.engine, kw, p.rtf, measured) + LOG_PATH))

    def _on_cleanup(self):
        self.cfg["cleanup_service"] = bool(self.cleanup_var.get())
        save_config(self.cfg)

    # --- корзина (задание 12): голоса и служебные файлы книг, 30 дней ---
    def _build_trash_card(self, page):
        ct = T.card(page)
        ct.pack(fill="x", padx=24, pady=(0, 14))
        bt = ctk.CTkFrame(ct, fg_color="transparent")
        bt.pack(fill="x", padx=18, pady=16)
        head = ctk.CTkFrame(bt, fg_color="transparent")
        head.pack(fill="x")
        T.section(head, _("Корзина")).pack(side="left")
        self.trash_clear_btn = T.button(head, _("Очистить корзину"), "x", kind="ghost", height=26,
                                        size=12, width=160, command=self._trash_clear)
        self.trash_clear_btn.pack(side="right")
        T.label(bt, _("Удалённые голоса и служебные файлы книг. Через 30 дней удаляются сами."),
                12, color=T.MUTED, anchor="w", justify="left", wraplength=640).pack(fill="x", pady=(2, 6))
        self.trash_rows = ctk.CTkFrame(bt, fg_color="transparent")
        self.trash_rows.pack(fill="x")
        self.trash_note = T.label(bt, "", 12, color=T.MUTED, anchor="w")
        self.trash_note.pack(fill="x", pady=(6, 0))
        self._refresh_trash()

    def _refresh_trash(self):
        from core import trash
        for w in self.trash_rows.winfo_children():
            w.destroy()
        its = trash.items()
        self.trash_clear_btn.configure(state="normal" if its else "disabled")
        if not its:
            T.label(self.trash_rows, _("Корзина пуста"), 12, color=T.MUTED, anchor="w").pack(fill="x")
            return
        kinds = {"voice": _("голос"), "service": _("служебные файлы книги")}
        for it in its:
            line = ctk.CTkFrame(self.trash_rows, fg_color="transparent")
            line.pack(fill="x", pady=2)
            when = time.strftime("%d.%m %H:%M", time.localtime(it.deleted))
            T.label(line, f"{it.title}", 13, bold=True, anchor="w").pack(side="left")
            T.label(line, f"  ·  {kinds.get(it.kind, it.kind)} · {when} · {_mb(it.size)}", 12,
                    color=T.MUTED, anchor="w").pack(side="left")
            T.button(line, _("Удалить навсегда"), "x", kind="ghost", height=26, size=12, width=150,
                     command=lambda it=it: self._trash_purge(it)).pack(side="right")
            T.button(line, _("Вернуть"), "refresh-cw", kind="ghost", height=26, size=12, width=100,
                     command=lambda it=it: self._trash_restore(it)).pack(side="right", padx=(0, 6))

    def _trash_restore(self, it):
        from core import trash
        replace = False
        if trash.conflicts(it):
            if not messagebox.askyesno(_("Вернуть"), _('«{0}» уже есть на прежнем месте. Заменить?').format(it.title)):
                return
            replace = True
        try:
            trash.restore(it, replace=replace)
            note = _('Возвращено: {0}').format(it.title)
        except FileNotFoundError:
            note = _("Папки книги нет - вернуть некуда")
        except Exception as e:
            log_problem("из корзины не вернулось", e, show=False)
            note = tr_msg(str(e))
        self.trash_note.configure(text=note)
        if it.kind == "voice":
            self.refresh_voices()
            self._refresh_voice_rows()
            self._vm_note_undo(note, None)
        else:
            self._service_line()
        self._refresh_trash()

    def _trash_purge(self, it):
        from core import trash
        if not messagebox.askyesno(_("Удалить навсегда"), _('Удалить «{0}» навсегда? Вернуть будет нельзя.').format(it.title)):
            return
        trash.purge(it)
        self.trash_note.configure(text=_('Удалено навсегда: {0}').format(it.title))
        self._refresh_trash()

    def _trash_clear(self):
        from core import trash
        its = trash.items()
        if not its or not messagebox.askyesno(_("Очистить корзину"), _('Удалить навсегда всё из корзины ({0})?').format(len(its))):
            return
        for it in its:
            trash.purge(it)
        self.trash_note.configure(text=_("Корзина пуста"))
        self._refresh_trash()

    def _pause(self, key: str) -> float:
        sl, _v = self._pause_vars[key]
        return round(sl.get() / 0.05) * 0.05

    def _on_pause(self, key: str, save: bool = True):
        _sl, val = self._pause_vars[key]
        val.configure(text=_('{0:.2f} с').format(self._pause(key)).replace(".", _dec()))
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
                text=_("Считается на процессоре: час звука - около суток счёта, книга - "
                     "несколько суток. Быстро на процессоре работают готовые голоса "
                     "Silero (страница «Голоса»)."),
                text_color=T.WARN)
        else:
            self.accel_hint.configure(text=_("Считается: ") + _(profiles.ACCEL_TITLE.get(accel, accel))
                                      + _(" · меняется в Настройках"), text_color=T.MUTED)
        self.show_estimate()

    def _nvidia_download(self):
        from core import cuda_pack
        self.nv_btn.configure(state="disabled", text=_("Скачиваю…"))

        def prog(got, total):
            pct = int(got * 100 / total) if total else 0
            self.ui(lambda: self.nv_btn.configure(text=_('Ускорение NVIDIA: {0}%').format(pct)))

        def work():
            try:
                cuda_pack.download(prog)
                self.ui(lambda: (self.nv_btn.configure(text=_("Готово")),
                                 self.nv_note.configure(text=_("Ускорение скачано. Перезапустите "
                                                             "Herald - озвучка пойдёт на видеокарте."),
                                                        text_color=T.OK)))
            except Exception as e:
                log_problem("ускорение NVIDIA не скачалось", e, show=False)
                self.ui(lambda: (self.nv_btn.configure(state="normal",
                                                       text=_("Скачать ускорение NVIDIA")),
                                 self.nv_note.configure(text=_('Не скачалось: {0}').format(tr_msg(str(e))),
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

    def _learn_book(self, dur: float, took: float, fresh: float | None = None):
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
        # 3.8: куски из кэша - не замер скорости. Ожидание считаем на секунду
        # НОВОГО звука; если нового почти не было - не запоминаем вовсе
        if fresh is not None and fresh < dur - 1:
            dur = fresh
        if not v.ready and dur >= 20:
            key = self._rtf_key()
            rw = self.cfg.setdefault("rtf_wall", {})
            w = took / dur
            for k_ in (f"{key}|{v.id}", f"{key}|*"):     # голос и режим целиком
                rw[k_] = round(w if k_ not in rw else 0.3 * rw[k_] + 0.7 * w, 3)
        save_config(self.cfg)
        self._cache_key = None              # книга теперь в кэше - пересчитать
        self.show_estimate()

    def _len(self) -> str:
        """Длина частей - исходное значение (в меню - перевод)."""
        v = self.len_var.get()
        return next((x for x in FILE_LENGTHS if v in (x, _(x))), "15 минут")

    def _fmt(self) -> str:
        v = self.fmt_var.get()
        return next((x for x in FORMATS if v in (x, _(x), self._fmt_label(x))), DEFAULT_FORMAT)

    def profile(self) -> profiles.Profile:
        return profiles.get(self.profile_var.get())

    def _quality(self) -> str:
        return self.quality_var.get() if hasattr(self, "quality_var") else profiles.DEFAULT_QUALITY

    def _prof_kwargs(self, p: profiles.Profile | None = None) -> dict:
        """Параметры движка профиля с режимом «Качество / скорость» (3.6)."""
        return profiles.engine_kwargs(p or self.profile(), self._quality())

    # --- менеджер голосов (3.7) ---
    # --- голоса: послушать, скачать, удалить (3.7; с 3.8 - на странице «Голоса») ---
    def _vm_listen_id(self, vid: str):
        from core import voice_store as VS
        v = VS.by_id(vid)
        if v is not None:
            return self._vm_listen(v)
        self.select_fx_voice(vid)                  # клон - фраза его звучанием
        self._fx_play()

    def _vm_download_id(self, vid: str):
        from core import voice_store as VS
        v = VS.by_id(vid)
        if v is not None:
            self._vm_download(v)

    def _vm_toggle_hidden(self, vid: str):
        h = self._hidden_voices()
        h.symmetric_difference_update({vid})
        self.cfg["hidden_voices"] = sorted(h)
        save_config(self.cfg)
        self.refresh_voices()
        self._refresh_voice_rows()

    def _vm_listen(self, v):
        """Образец: Silero и скачанные - синтез фразы; Piper без загрузки - официальный."""
        from core import voice_store as VS
        if v.engine == "piper" and not VS.installed(v):
            self.vm_note.configure(text=_("Беру официальный образец Piper…"))

            def work():
                try:
                    path = VS.sample(v)
                    self.ui(lambda: self.vm_note.configure(text=_('Образец {0}: играет').format(_(v.title))))
                    p = Player()
                    if p.load(path):
                        p.play()
                        while p.playing:
                            time.sleep(0.1)
                except Exception as e:
                    self.ui(lambda: self.vm_note.configure(text=tr_msg(str(e))))
            threading.Thread(target=work, daemon=True).start()
            return
        text = self.FX_PHRASE.get(v.language, self.FX_PHRASE["ru"])
        if v.language == "ru":
            text = text.replace("+", "")
        self._speak_job("vm", text, v.engine, {}, v.id, note=self.vm_note)

    def _vm_download(self, v, mirror=None):
        """Скачать голос; ход - внизу «Голосов» и (mirror) в строке другой страницы."""
        from core import voice_store as VS
        if self._vm_busy is not None:
            return
        size = _mb(VS.download_size(v))
        what = _("пакет Vosk (3 голоса)") if v.engine == "vosk" else _('голос {0}').format(_(v.title))
        lic = (VS.PIPER_LICENSE.format(lic=v.license) if v.engine == "piper"
               else _('Лицензия: {0}. Источник: alphacephei.com.').format(v.license))
        if not messagebox.askokcancel(_("Скачать голос"), _('Скачать {0} ({1})?\n\n{2}').format(what, size, lic)):
            return
        self._vm_busy = v
        self.vm_prog.pack(fill="x", pady=(8, 0), before=self.vm_note)
        self.vm_prog.set(0)

        def prog(fr, msg):
            self.ui(lambda: (self.vm_prog.set(fr), self.vm_note.configure(text=msg),
                             mirror.configure(text=f"{msg} · {int(fr * 100)} %") if mirror else None))

        def work():
            err = None
            try:
                VS.download(v, progress=prog)
            except Exception as e:
                err = e
                log_problem(f"голос {v.id} не скачался", e, show=False)

            def end():
                self._vm_busy = None
                self.vm_prog.pack_forget()
                self.vm_note.configure(text=tr_msg(str(err)) if err else _('Готово: {0} скачан').format(what))
                if mirror is not None:
                    mirror.configure(text=self.vm_note.cget("text"))
                self.refresh_voices()
                self._refresh_voice_rows()
            self.ui(end)
        threading.Thread(target=work, daemon=True).start()

    def _vm_delete(self, v):
        """«Удалить…». Piper - файл голоса с диска. Vosk - один файл модели на три
        голоса: удаление одного голоса его скрывает, а когда скрыты все три -
        предлагаем удалить сам пакет (3.8)."""
        from core import voice_store as VS
        if v.engine == "vosk":
            vosk = [x.id for x in VS.CATALOG if x.engine == "vosk"]
            if not messagebox.askyesno(_("Удалить голос"), _('Скрыть голос {0}? Он входит в пакет Vosk вместе с другими - файл пакета останется, пока нужен им.').format(self._vname(v.id, v.title))):
                return
            h = self._hidden_voices() | {v.id}
            self.cfg["hidden_voices"] = sorted(h)
            save_config(self.cfg)
            note = _('Скрыт: {0}').format(self._vname(v.id, v.title))
            if all(x in h for x in vosk) and VS.installed(v) and messagebox.askyesno(
                    _("Удалить пакет Vosk"),
                    _('Все три голоса Vosk скрыты. Удалить пакет Vosk ({0}) с диска?\nЕго можно будет скачать снова.').format(_mb(VS.size_on_disk(v)))):
                VS.delete(v)
                self.cfg["hidden_voices"] = sorted(h - set(vosk))   # скачают снова - видны
                save_config(self.cfg)
                note = _("Удалено: пакет Vosk")
            self.vm_note.configure(text=note)
        else:
            what = _('голос {0}').format(self._vname(v.id, v.title))
            if not messagebox.askyesno(_("Удалить голос"), _('Удалить {0} с диска?\nЕго можно будет скачать снова.').format(what)):
                return
            VS.delete(v)
            self.vm_note.configure(text=_('Удалено: {0}').format(what))
        self.refresh_voices()
        self._refresh_voice_rows()

    def _on_say_clones(self):
        self.cfg["say_clones"] = bool(self.say_clones_var.get())
        save_config(self.cfg)
        if hasattr(self, "say_mode"):
            self._say_on_kind(self.say_mode.get() if self._say_clones() else _("Быстрый"),
                              remember=False)

    def _on_audio_out(self):
        from core import audio_out
        v = self.out_var.get()
        name = audio_out.SYSTEM if v == _("Как в системе") else v
        self.cfg["audio_out"] = name
        save_config(self.cfg)
        audio_out.set_output(name)

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
        self.cfg.update(len=self._len(), fmt=self._fmt(),
                        voice=self.current_voice())
        save_config(self.cfg)

    # ================= голоса =================
    def _all_voices(self) -> list[_V]:
        """Все голоса, и скрытые тоже (у них hidden=True): обученные и клоны F5, голоса
        Qwen (если Qwen есть на этом Mac), готовые Silero и скачанные Vosk / Piper."""
        out = []
        names = self.cfg.get("voice_names") or {}
        hidden = self._hidden_voices()
        vdir = default_voices_dir()
        f5ids = set()
        try:
            for v in get_engine(clone_engine()).list_voices():
                f5ids.add(v.id)
                mine = os.path.isfile(os.path.join(vdir, v.id + ".f5model"))
                out.append(_V(v.id, names.get(v.id) or v.id, v.language, False, clone_engine(),
                              mine=mine, hidden=v.id in hidden))
        except Exception as e:
            log_problem("не удалось прочитать список голосов", e)
        if self._qwen_ok():
            try:
                from core.engines.mlxaudio import own_voices
                out += [_V(n, names.get(n) or n, "ru", False, "mlxaudio", qwen=True,
                           hidden=n in hidden) for n in own_voices() if n not in f5ids]
            except Exception as e:
                log_problem("голоса Qwen не прочитались", e, show=False)
        out += [_V(v.id, names.get(v.id) or v.title, v.language, True, "silero", v.gender,
                   hidden=v.id in hidden) for v in SILERO_VOICES]
        # быстрые голоса Vosk / Piper - выбор Андрея 3.6, только скачанные (core/voice_store)
        try:
            from core.engines import fast
            out += [_V(v.id, names.get(v.id) or v.title, v.language, True,
                       fast.engine_of(v.id), v.gender, hidden=v.id in hidden)
                    for v in fast.enabled_voices()]
        except Exception as e:
            log_problem("быстрые голоса не прочитались", e, show=False)
        return out

    def _qwen_ok(self) -> bool:
        """Есть ли Qwen ru-stress на этом компьютере (проверка файлов - один раз)."""
        if not hasattr(self, "_qwen_cache"):
            try:
                self._qwen_cache = profiles.available(profiles.PROFILES["qwen"])
            except Exception:
                self._qwen_cache = False
        return self._qwen_cache

    def _hidden_voices(self) -> set:
        from core.voice_store import HIDDEN_DEFAULT
        h = self.cfg.get("hidden_voices")
        return set(HIDDEN_DEFAULT if h is None else h)

    def _voice_groups(self, vs: list[_V], label=None) -> list:
        """[(«Мои голоса · 2», [подписи]), …] - для выпадающих списков."""
        label = label or (lambda v: v.label())
        out = []
        for g, title in GROUPS:
            labs = [label(v) for v in vs if v.group() == g]
            out.append((f"{_(title)} · {len(labs)}", labs))
        return out

    def refresh_voices(self):
        # один порядок везде (задание 12): «Мои голоса», русские, английские - по алфавиту;
        # скрытые в списках выбора не показываются
        self._voices = _sorted_voices([v for v in self._all_voices() if not v.hidden])
        lang = getattr(self.est, "language", None)
        if lang:
            # голоса языка книги - выше (группа «Мои голоса» остаётся первой)
            self._voices.sort(key=lambda v: (not v.mine and v.language != lang))
        labels = [v.label() for v in self._voices] or [""]
        _grouped_menu(self.voice_menu, self._voice_groups(self._voices))
        want = self.cfg.get("voice")
        fits = [v for v in self._voices if not lang or v.language == lang] or self._voices
        cur = next((v.label() for v in fits if v.id == want), None)
        now = next((v for v in self._voices if v.label() == self.voice_var.get()), None)
        # голос другого языка книге не подходит: берём сохранённый подходящий,
        # иначе первый подходящий (первый - из «Моих голосов», 3.3.11)
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
            self.voice_hint.configure(text=_("Голосов пока нет - добавьте на странице «Голоса»."))
        else:
            lang = getattr(self.est, "language", None)
            names = {"ru": _("русском"), "en": _("английском")}
            warn = (_(' Книга на {0}, а голос - на {1}: выйдет с акцентом.').format(names.get(lang, lang), names.get(v.language, v.language))
                    if lang and lang != v.language else "")
            kind = (_("готовый голос, без клонирования") if v.ready
                    else _("голос Qwen - читает движок Qwen ru-stress") if v.qwen
                    else _("обучен на записях чтеца") if v.mine else _("клон по образцу"))
            self.voice_hint.configure(text=kind + "." + warn,
                                      text_color=T.WARN if warn else T.MUTED)
        if self.profile_chip:
            T.set_chip(self.profile_chip, _(profiles.SILERO.title if v is not None and v.ready
                                            else profiles.PROFILES["qwen"].title
                                            if v is not None and v.qwen
                                            else self.profile().title), "acc")
        if remember:
            self._remember()
        self._srv_sync()
        self.show_estimate()

    def _refresh_voice_rows(self):
        """Список голосов (задание 12): группы «Мои голоса», «Русские», «Английские» с
        числом голосов, внутри - по алфавиту. У строки «Прослушать», «Скачать» (если не
        скачан), «✎» - своё имя, «⋯» - скрыть / показать / удалить. Скрытые - только при
        «Показать скрытые», серым. Щелчок по строке выбирает голос для озвучки и
        открывает справа его звучание."""
        from core import voice_store as VS
        for w in self.voice_rows.winfo_children():
            w.destroy()
        self._voice_row_widgets = {}
        show_hidden = bool(getattr(self, "show_hidden_var", None) and self.show_hidden_var.get())
        fxdir = default_voices_dir()
        chosen = self.current_voice()
        eng_title = {"silero": "Silero", "vosk": "Vosk", "piper": "Piper"}
        vosk_size = _mb(VS.VOSK_SIZE)
        rows = []                       # (группа, имя для сортировки, функция-строка)
        for v in self._all_voices():
            if v.ready or (v.hidden and not show_hidden):
                continue
            meta = [_("обучен на записях чтеца") if v.mine else "",
                    _(LANG_MARK.get(v.language, v.language)),
                    "Qwen" if v.qwen else ("" if v.mine else _("клон"))]
            fx = load_fx(fxdir, v.id)
            if not fx.is_neutral():
                meta.append(_("звучание настроено"))
            if v.hidden:
                meta.append(_("скрыт"))
            if v.id == chosen and not v.hidden:
                meta.append(_("выбран для озвучки"))
            menu = [("rename", _("Переименовать…")),
                    ("show", _("Показать")) if v.hidden else ("hide", _("Скрыть")),
                    ("trash", _("Удалить…"))]
            rows.append((v.group(), _(v.title).lower(), lambda v=v, meta=meta, menu=menu: self._voice_row(
                v.id, v.title, v.id, meta, usable=not v.hidden, chosen=v.id == chosen and not v.hidden,
                listen=True, download=False, menu=menu)))
        hidden = self._hidden_voices()
        for c in VS.CATALOG:
            have = VS.installed(c)
            hid = c.id in hidden
            if hid and not show_hidden:
                continue
            meta = [_(GENDER.get(c.gender, "")), _(LANG_MARK.get(c.language, c.language)),
                    eng_title[c.engine]]
            if c.engine == "silero":
                meta.append(_("встроен"))
            elif c.engine == "vosk":
                meta.append(_('входит в пакет Vosk {0}').format(vosk_size))
            else:
                meta.append(_mb(VS.size_on_disk(c) if have else VS.download_size(c)))
            if not have:
                meta.append(_("не скачан"))
            if hid:
                meta.append(_("скрыт"))
            if have and c.id == chosen:
                meta.append(_("выбран для озвучки"))
            menu = [("rename", _("Переименовать…")),
                    ("show", _("Показать")) if hid else ("hide", _("Скрыть"))]
            if c.engine != "silero" and have:
                menu.append(("delete", _("Удалить…")))
            g = "en" if c.language == "en" else "ru"
            rows.append((g, self._vname(c.id, c.title).lower(), lambda c=c, meta=meta, menu=menu, have=have, hid=hid: self._voice_row(
                c.id, self._vname(c.id, c.title), c.title, meta,
                usable=have and not hid, chosen=have and c.id == chosen,
                listen=have or c.engine == "piper", download=not have, menu=menu)))
        order = {g: i for i, (g, _t) in enumerate(GROUPS)}
        rows.sort(key=lambda r: (order[r[0]], r[1]))
        for g, title in GROUPS:
            mine = [r for r in rows if r[0] == g]
            if not mine and g != "mine":
                continue
            T.section(self.voice_rows, f"{_(title)} · {len(mine)}").pack(fill="x", padx=14, pady=(12, 4))
            if not mine:
                T.label(self.voice_rows, _("пока ни одного - голос, обученный на записях чтеца, "
                                           "появится здесь"), 12, color=T.MUTED, anchor="w",
                        justify="left", wraplength=520).pack(fill="x", padx=14)
            for r in mine:
                r[2]()

    def _vname(self, vid: str, title: str) -> str:
        """Имя голоса для показа: своё (✎), иначе исходное (переведённое)."""
        return (self.cfg.get("voice_names") or {}).get(vid) or _(title)

    def _voice_row(self, vid, name, orig, meta, *, usable, chosen, listen, download, menu):
        row = ctk.CTkFrame(self.voice_rows, fg_color="transparent", corner_radius=8,
                           cursor=osutil.HAND if usable else "")
        row.pack(fill="x", padx=8, pady=1)
        row.grid_columnconfigure(1, weight=1)
        ic = ctk.CTkLabel(row, text="", image=T.icon("check" if chosen else "mic",
                                                     T.ACCENT if chosen else T.MUTED, 18))
        ic.grid(row=0, column=0, rowspan=2, padx=(10, 10), pady=(8, 0), sticky="n")
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.grid(row=0, column=1, sticky="ew", pady=(6, 0))
        nm = T.label(top, name, 14, bold=True, anchor="w",
                     color=T.TEXT if usable else T.MUTED)
        nm.pack(side="left")
        T.button(top, "✎", None, kind="ghost", height=22, width=26, size=13,
                 command=lambda: self._rename_voice(vid, orig)).pack(side="left", padx=(4, 0))
        if name != _(orig) and name != orig:
            T.label(top, _(orig), 11, color=T.FAINT, anchor="w").pack(side="left", padx=(6, 0))
        sub = T.label(row, "  ·  ".join(x for x in meta if x), 12, color=T.MUTED, anchor="w")
        sub.grid(row=1, column=1, sticky="ew", pady=(0, 6))
        acts = ctk.CTkFrame(row, fg_color="transparent")
        acts.grid(row=0, column=2, rowspan=2, padx=(4, 8))
        if listen:                              # ▶ - «Послушать» (значок: строка узкая)
            T.button(acts, "", "play", kind="ghost", height=28, width=32,
                     command=lambda: self._vm_listen_id(vid)).pack(side="left")
        if download:
            T.button(acts, _("Скачать"), "upload", kind="default", height=28, size=12, width=100,
                     command=lambda: self._vm_download_id(vid)).pack(side="left", padx=(4, 0))
        T.button(acts, "⋯", None, kind="ghost", height=28, width=30, size=14,
                 command=lambda: self._voice_menu(vid, orig, menu, acts)).pack(side="left",
                                                                               padx=(4, 0))
        if usable:
            for w in (row, ic, nm, sub):
                w.bind("<Button-1>", lambda e: self.pick_voice(vid))
                w.bind("<Enter>", lambda e: self._voice_hover(vid, True))
                w.bind("<Leave>", lambda e: self._voice_hover(vid, False))
        self._voice_row_widgets[vid] = row
        row._herald_menu = menu                        # для проверок
        self._voice_hover(vid, False)

    def _voice_menu(self, vid, orig, menu, anchor):
        """«⋯» строки: переименовать, скрыть / показать, удалить (с подтверждением)."""
        import tkinter as tk
        m = tk.Menu(anchor, tearoff=0)
        for act, title in menu:
            m.add_command(label=title, command=lambda a=act: self._voice_action(vid, orig, a))
        m.tk_popup(anchor.winfo_rootx(), anchor.winfo_rooty() + anchor.winfo_height())

    def _voice_action(self, vid: str, orig: str, act: str):
        from core import voice_store as VS
        if act == "rename":
            return self._rename_voice(vid, orig)
        if act in ("hide", "show"):
            return self._vm_toggle_hidden(vid)
        if act == "trash":
            return self._voice_trash(vid)
        if act == "delete":
            v = VS.by_id(vid)
            if v is not None:
                self._vm_delete(v)

    def _voice_files(self, v: _V) -> list[str]:
        """Файлы голоса (без весов обучения - папка весов остаётся на месте)."""
        from core.voicefx import fx_path
        vd = default_voices_dir()
        d = os.path.join(vd, "qwen") if v.qwen else vd
        exts = (".wav", ".txt", ".src.json", ".f5model", ".gen.json", ".model")
        out = [os.path.join(d, v.id + e) for e in exts if os.path.exists(os.path.join(d, v.id + e))]
        fx = fx_path(vd, v.id)                # поправки звучания - всегда в общей папке голосов
        return out + ([fx] if os.path.exists(fx) else [])

    def _voice_trash(self, vid: str, ask: bool = True):
        """«Удалить…» клона, обученного, Qwen: файлы голоса - в корзину на 30 дней."""
        from core import trash
        v = self._voice_obj(vid)
        if v is None or v.ready:
            return
        name = self._vname(v.id, v.title)
        if ask and not messagebox.askyesno(
                _("Удалить голос"), _('Удалить голос «{0}»? Его можно вернуть из корзины 30 дней '
                                      '(Настройки › Корзина).').format(name)):
            return
        item = trash.put(self._voice_files(v), "voice", name)
        if self._fx_voice == vid:
            self._fx_voice = None
            self.fx_title.configure(text=_("Выберите голос слева"))
            self.fx_src.configure(text="")
            self._fx_enable(False)
        self.refresh_voices()
        self._refresh_voice_rows()
        self._vm_note_undo(_('Удалено: {0}').format(name), item)
        if hasattr(self, "trash_rows"):
            self._refresh_trash()

    def _vm_note_undo(self, text: str, item):
        """Строка внизу списка голосов: «Удалено: … · Вернуть»."""
        self.vm_note.configure(text=text)
        btn = getattr(self, "vm_undo", None)
        if btn is not None and btn.winfo_exists():
            btn.destroy()
        if item is None:
            return
        self.vm_undo = T.button(self.vm_note.master, _("Вернуть"), "refresh-cw", kind="ghost",
                                height=26, size=12, width=110,
                                command=lambda: self._trash_restore(item))
        self.vm_undo.pack(anchor="w", pady=(4, 0))

    def _rename_voice(self, vid: str, orig: str):
        """«✎»: своё имя голоса во всех списках; исходное - мелко серым рядом."""
        dlg = ctk.CTkToplevel(self)
        dlg.title(_("Имя голоса"))
        T.label(dlg, _('Как показывать голос «{0}»:').format(_(orig)), 13, anchor="w").pack(
            fill="x", padx=16, pady=(14, 6))
        ent = ctk.CTkEntry(dlg, width=300, font=T.font(14))
        ent.insert(0, (self.cfg.get("voice_names") or {}).get(vid, ""))
        ent.pack(padx=16)
        btns = ctk.CTkFrame(dlg, fg_color="transparent")
        btns.pack(fill="x", padx=16, pady=(10, 14))

        def save(name: str):
            names = dict(self.cfg.get("voice_names") or {})
            name = name.strip()[:60]
            if name and name != _(orig) and name != orig:
                names[vid] = name
            else:
                names.pop(vid, None)
            self.cfg["voice_names"] = names
            save_config(self.cfg)
            dlg.destroy()
            self.refresh_voices()
            self._refresh_voice_rows()
            if getattr(self, "_fx_voice", None) == vid:
                self.fx_title.configure(text=self._vname(vid, orig))
        T.button(btns, _("Сохранить"), "save", kind="primary", height=32, width=130,
                 command=lambda: save(ent.get())).pack(side="left")
        T.button(btns, _("Исходное имя"), "refresh-cw", kind="ghost", height=32, width=150,
                 command=lambda: save("")).pack(side="left", padx=6)
        ent.bind("<Return>", lambda e: save(ent.get()))
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg._save = save                                   # для проверок
        self._rename_dlg = dlg
        ent.focus_set()

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
        self.fx_title.configure(text=self._vname(vid, v.title) if v else vid)
        src_txt = _("готовый голос Silero") if v and v.ready else ""
        sp = os.path.join(default_voices_dir(), vid + ".src.json")
        if os.path.exists(sp):
            try:
                src_txt = _("образец: ") + os.path.basename(
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
        self._fx_more_show(v, fx)
        self._refresh_voice_rows()

    DENOISE = (("нет", 0.0), ("мягкое", 15.0), ("сильное", 100.0))

    def _fx_more_show(self, v, fx: VoiceFX):
        """«Дополнительно»: свои веса / выборка Qwen - показ; шумоподавление - выбор."""
        info = []
        if v is not None and v.mine:
            info.append(_("Свои веса: обучен на записях чтеца (F5, дообучение)."))
        if v is not None and v.qwen:
            kw = profiles.PROFILES["qwen"].kwargs
            info.append(_("Движок: Qwen ru-stress, выборка: temperature {0} · top_p {1}.").format(
                str(kw.get("temperature", 1.0)).replace(".", _dec()),
                str(kw.get("top_p", 0.95)).replace(".", _dec())))
        if v is not None and not v.mine and not v.qwen and not v.ready:
            info.append(_("Клон по образцу: общие веса модели, голос - из образца."))
        self.fx_info.configure(text=" ".join(info))
        from core.denoise import dfn_binary
        try:
            have_dfn = bool(dfn_binary())
        except Exception:
            have_dfn = False
        if have_dfn and v is not None and not v.qwen:
            dn = fx.denoise or 0.0
            near = min(self.DENOISE, key=lambda x: abs(x[1] - dn))
            self.dn_var.set(_(near[0]))
            self.dn_box.pack(fill="x")
        else:
            self.dn_box.pack_forget()

    def _fx_denoise(self, label: str):
        """Шумоподавление голоса - сразу в его поправки (остальные поправки не трогаются)."""
        if not self._fx_voice:
            return
        val = next((x[1] for x in self.DENOISE if label in (x[0], _(x[0]))), 0.0)
        fx = load_fx(default_voices_dir(), self._fx_voice)
        fx.denoise = val
        save_fx(default_voices_dir(), self._fx_voice, fx)
        self.fx_note.configure(text=_('Сохранено для «{0}»').format(self._vname(self._fx_voice, self._fx_voice)))
        if hasattr(self, "say_fx_line"):
            self._say_fx_show()
        self._refresh_voice_rows()

    def _fx_enable(self, on: bool):
        st = "normal" if on else "disabled"
        for sl, _l, _u in self._fx_vars.values():
            sl.configure(state=st)
        for b in (self.fx_play_btn, self.fx_save_btn):
            b.configure(state=st)
        if not on and hasattr(self, "fx_more"):
            self.fx_info.configure(text="")
            self.dn_box.pack_forget()

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
            lbl.configure(text=_("как есть") if abs(v - 1) < 0.01
                          else f"{v:.2f}".replace(".", _dec()) + unit)
        else:
            v = round(sl.get(), 1)
            lbl.configure(text=_("нет") if v == 0 else f"{v:+.1f}".replace(".", _dec()) + unit)

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
        elif v and v.qwen:
            engine, kw = "mlxaudio", self._prof_kwargs(profiles.PROFILES["qwen"])
            text = self.FX_PHRASE["ru"]
        else:
            kw = dict(self._prof_kwargs(), speed=fx.speed)
            if lang != "ru":
                kw.pop("variant", None)
            engine, text = self._clone_engine_for(), self.FX_PHRASE.get(lang, self.FX_PHRASE["ru"])

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
                [(self.fx_play_btn, "fx", _("Прослушать"), "play", "default")] +
                [(b, "prof:" + key, _("Послушать"), "play", "ghost")
                 for key, b in self._prof_btns.items()]):
            if k == kind:
                btn.configure(text=_("Стоп"), state="normal")
                T.restyle(btn, "danger", "square")
            else:
                btn.configure(text=title)
                T.restyle(btn, style, icon)
                ok = not self.running and not (btn is self.fx_play_btn and not self._fx_voice)
                btn.configure(state="normal" if ok else "disabled")
        if hasattr(self, "say_stop_btn"):
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
                txt = tr_msg(S.status_text(st))
                if not live.gate and st["state"] == "wait" and live.speed.rtf > 1.0:
                    txt += _(" · возможны паузы")
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
        show(_("Готовлю голос…"))
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
                    show(_('Не вышло: {0}').format(tr_msg(str(err))))
                elif live.first_sound is not None and not live.stopped:
                    show(_('Готово · первый звук через {0:.1f} с').format(live.first_sound)
                         .replace(".", _dec()))
                elif live.stopped:
                    show(_("Остановлено"))
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
        """Клон F5 для языка (голоса Qwen сюда не берутся: их читает только Qwen)."""
        v = self._voice_obj()
        if v is not None and not v.ready and not v.qwen and v.language == lang:
            return v.id
        for x in self._voices:
            if not x.ready and not x.qwen and x.language == lang:
                return x.id
        return v.id if v is not None and not v.ready and not v.qwen else None

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
        """Текст книги так, как он уйдёт в модель: числа словами, ударения; слова,
        где RUAccent и silero-stress разошлись, подсвечены (3.8). Текст не правится, но
        спорное слово можно поправить для этой книги (задание 12, core/bookfix)."""
        if not self.book_path:
            messagebox.showwarning(_("Нет книги"), _("Перетащите книгу в окно."))
            return
        path, minutes = self.book_path, self.file_minutes()
        self.readtext_btn.configure(state="disabled", text=_("Готовлю текст…"))

        def work():
            try:
                from core import pipeline as P
                saved = P.read_text_path(path)
                # сохранённый текст годится, если он новее книги и сделан с той же длиной частей
                fresh = (os.path.exists(saved) and os.path.getmtime(saved) >= os.path.getmtime(path)
                         and abs(minutes - float(self.cfg.get("read_text_minutes", minutes))) < 0.01)
                if fresh:
                    text = open(saved, encoding="utf-8").read()
                else:
                    text = P.prepare_read_text(path, ConvertOptions(minutes_per_file=minutes))
                    self.cfg["read_text_minutes"] = minutes
                disputed = P.read_disputed(path)
                self.ui(lambda: self._read_text_window(text, disputed, path))
            except Exception as e:
                log_problem("«как будет прочитано» не получилось", e, show=False)
                self.ui(lambda: messagebox.showerror(_("Ошибка"), str(e)))
            finally:
                self.ui(lambda: self.readtext_btn.configure(state="normal",
                                                            text=_("Как будет прочитано")))
        threading.Thread(target=work, daemon=True).start()

    def _read_text_window(self, text: str, disputed=(), book: str | None = None):
        from core import bookfix
        win = ctk.CTkToplevel(self)
        win.title(_("Как будет прочитано"))
        win.geometry("760x680")
        book = book or self.book_path
        want = {bookfix.key(w) for w in disputed}
        note = _("Так текст уйдёт в модель: числа словами, ударение - знак над гласной.")
        if want:
            note += " " + _("Подсвечены спорные слова ({0}): щелчок - выбрать ударение для этой "
                            "книги.").format(len(want))
        T.label(win, note, 12, color=T.MUTED, anchor="w", justify="left", wraplength=700).pack(
            fill="x", padx=16, pady=(12, 2))
        fixed_note = T.label(win, "", 12, color=T.OK, anchor="w")
        fixed_note.pack(fill="x", padx=16, pady=(0, 6))
        box = ctk.CTkTextbox(win, font=T.font(15), wrap="word", border_width=1)
        box.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        inner = getattr(box, "_textbox", box)
        inner.tag_configure("disputed", background=SAY_HL_BG, foreground=SAY_HL_FG, underline=True)
        inner.tag_configure("fixed", background="#CDEFD2", foreground=SAY_HL_FG, underline=True)

        def render():
            from core.stress import to_combining
            fixes = bookfix.load(book) if book else {}
            body = to_combining(bookfix.apply(text, fixes))
            box.configure(state="normal")
            box.delete("1.0", "end")
            box.insert("1.0", body)
            for m in re.finditer(r"[А-Яа-яЁё\u0301]+", body):
                k = bookfix.key(m.group())
                if k in fixes:
                    inner.tag_add("fixed", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
                elif k in want:
                    inner.tag_add("disputed", f"1.0+{m.start()}c", f"1.0+{m.end()}c")
            box.configure(state="disabled")
            fixed_note.configure(text=_('Поправлено слов: {0} · применится при следующей озвучке').format(
                len(fixes)) if fixes else "")
            win._disputed = len(inner.tag_ranges("disputed")) // 2              # для проверок
            win._fixed = len(fixes)

        def pick(event):
            """Щелчок по спорному слову: меню «Ударение: …» по каждой гласной."""
            import tkinter as tk
            idx = inner.index(f"@{event.x},{event.y}")
            tags = inner.tag_names(idx)
            if "disputed" not in tags and "fixed" not in tags:
                return
            word = inner.get(f"{idx} wordstart", f"{idx} wordend")
            a0 = inner.index(f"{idx} wordstart")
            # «wordstart» Tk не считает знак ударения частью слова - берём весь отрезок тега
            for tag in ("disputed", "fixed"):
                rng = inner.tag_prevrange(tag, f"{idx}+1c")
                if rng and inner.compare(rng[0], "<=", idx) and inner.compare(idx, "<", rng[1]):
                    word = inner.get(rng[0], rng[1])
                    break
            plain = word.replace("\u0301", "")
            m = tk.Menu(inner, tearoff=0)
            for i in bookfix.variants(plain):
                shown = plain[:i + 1] + "\u0301" + plain[i + 1:]
                m.add_command(label=_('Ударение: {0}').format(shown),
                              command=lambda i=i: (bookfix.set_fix(book, plain, i), render()))
            m.add_separator()
            m.add_command(label=_("Как предложила программа"),
                          command=lambda: (bookfix.set_fix(book, plain, None), render()))
            m.tk_popup(event.x_root, event.y_root)
            win._menu = m                                                       # для проверок
        for tag in ("disputed", "fixed"):
            inner.tag_bind(tag, "<Button-1>", pick)
            inner.tag_bind(tag, "<Enter>", lambda e: inner.configure(cursor=osutil.HAND))
            inner.tag_bind(tag, "<Leave>", lambda e: inner.configure(cursor=""))
        render()
        win.bind("<Escape>", lambda e: win.destroy())
        win._box = box                                                          # для проверок
        win._render = render
        win._pick = pick
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
        head.grid(row=0, column=0, columnspan=2, sticky="ew", padx=24, pady=(22, 14))
        T.label(head, _("Прочитать вслух"), 22, bold=True, anchor="w").pack(side="left")

        left = T.card(page)
        left.grid(row=1, column=0, sticky="nsew", padx=(24, 8), pady=(0, 20))
        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(12, 6))
        T.button(bar, _("Из буфера"), "copy", kind="ghost", height=30, size=13, width=120,
                 command=self._say_paste).pack(side="left")
        T.button(bar, _("Открыть файл…"), "folder-open", kind="ghost", height=30, size=13,
                 width=150, command=self._say_open).pack(side="left", padx=6)
        T.button(bar, "", "x", kind="ghost", height=30, width=34,                    # очистить
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
        # одна кнопка ▶ / ❚❚ (3.8); пробел - то же самое, если курсор не в тексте
        self.say_btn = T.button(ctrl, _("Прочитать"), "play", kind="primary", height=34, size=13,
                                width=150, command=self._say_toggle)
        self.say_btn.pack(side="left")
        self.say_stop_btn = T.button(ctrl, "", "square", height=34, width=44,
                                     command=self._say_stop)
        self.say_stop_btn.pack(side="left", padx=(6, 0))
        self.say_save_btn = T.button(ctrl, "", "save", kind="ghost", height=34, width=44,
                                     command=self._say_save)              # сохранить в mp3
        self.say_save_btn.configure(state="disabled")
        self.say_save_btn.pack(side="right")
        # строка состояния - отдельной строкой: между кнопками она обрезалась (3.6)
        self.say_note = T.label(left, "", 12, color=T.MUTED, anchor="w")
        self.say_note.pack(fill="x", padx=16, pady=(0, 10))

        right = T.card(page)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 24), pady=(0, 20))
        r = ctk.CTkFrame(right, fg_color="transparent")
        r.pack(fill="both", expand=True, padx=18, pady=16)
        ctk.CTkFrame(r, width=290, height=0, fg_color="transparent").pack()
        T.section(r, _("Голос")).pack(fill="x")
        # «Быстрый / Клон» - только если клоны показаны (Настройки, 3.6)
        self.say_mode_box = ctk.CTkFrame(r, fg_color="transparent")
        self.say_mode_box.pack(fill="x")
        self.say_mode = ctk.CTkSegmentedButton(
            self.say_mode_box, values=[_("Быстрый"), _("Клон")], height=30, font=T.font(13),
            selected_color=T.ACCENT, selected_hover_color=T.ACCENT_HOVER,
            command=lambda v: self._say_on_kind(v))
        self.say_mode.pack(fill="x", pady=(6, 4))
        self.say_voice_var = ctk.StringVar(value="")
        self.say_voice_menu = ctk.CTkOptionMenu(r, variable=self.say_voice_var, values=[""],
                                                height=30, font=T.font(13),
                                                dropdown_font=T.font(13),
                                                command=lambda *_a: self._say_on_voice())
        self.say_voice_menu.pack(fill="x")
        # «Сразу / Без пауз» - только для клона: быстрые голоса и так без пауз
        self.say_start_box = ctk.CTkFrame(r, fg_color="transparent")
        self.say_start_box.pack(fill="x")
        T.section(self.say_start_box, _("Начало")).pack(fill="x", pady=(14, 0))
        sv = self.cfg.get("say_start")
        self.say_start_var = ctk.StringVar(value=sv if sv in ("Сразу", "Без пауз") else "Сразу")
        for val, text in (("Сразу", _("Начать сразу (между фразами возможны паузы)")),
                          ("Без пауз", _("Подождать и читать без остановок"))):
            ctk.CTkRadioButton(self.say_start_box, text=text, value=val,
                               variable=self.say_start_var, font=T.font(13),
                               fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                               border_color=T.BORDER_STRONG, text_color=T.TEXT,
                               command=self._say_on_start).pack(anchor="w", pady=(6, 0))
        # «Звучание» - одно место, страница «Голоса» (задание 12); здесь - что сохранено
        self.say_sound_head = T.section(r, _("Звучание"))
        self.say_sound_head.pack(fill="x", pady=(14, 0))
        self.say_fx_line = T.label(r, "", 12, color=T.MUTED, anchor="w", justify="left",
                                   wraplength=290)
        self.say_fx_line.pack(fill="x", pady=(4, 0))
        self.say_fx_link = T.button(r, _("Звучание голоса ›"), None, kind="ghost", height=26,
                                    size=12, width=10,
                                    command=lambda: self._open_voice_fx(self._say_voice()))
        self.say_fx_link.configure(anchor="w")
        self.say_fx_link.pack(anchor="w", pady=(2, 0))
        self.say_forecast = T.label(r, "", 12, color=T.MUTED, anchor="w", justify="left",
                                    wraplength=290)
        self.say_forecast.pack(fill="x", pady=(14, 0))
        self._say_spans: list = []
        self._say_offset = 0
        self._say_from = 0
        self.say_mode.set(_(self.cfg.get("say_mode")
                            if self.cfg.get("say_mode") in ("Быстрый", "Клон") else "Быстрый"))
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
        for label, act in ((_("Вырезать"), "cut"), (_("Копировать"), "copy"), (_("Вставить"), "paste"),
                           (None, None), (_("Выделить всё"), "all")):
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
        self.say_count.configure(text=_('{0:,} знаков').format(n).replace(",", " ") if n else "")
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
            self.say_note.configure(text=_("В буфере нет текста"))

    def _say_open(self):
        path = filedialog.askopenfilename(
            filetypes=[(_("Тексты и книги"), "*.txt *.md *.docx *.fb2 *.epub *.pdf *.zip"),
                       (_("Все файлы"), "*.*")])
        if not path:
            return
        try:
            from core.readers import read_document
            doc = read_document(path)
            text = doc.text or "\n\n".join(
                (f"{c.title}\n\n" if getattr(c, "title", "") else "") + c.text
                for c in (doc.chapters or []))
        except Exception as e:
            messagebox.showerror(_("Не открылось"), str(e))
            return
        self.say_box.delete("1.0", "end")
        self.say_box.insert("1.0", text.strip())
        self._say_text_changed()

    def _say_clear(self):
        self._say_stop()
        self.say_box.delete("1.0", "end")
        self._say_text_changed()

    # --- голос ---
    def _say_clones(self) -> bool:
        return bool(self.cfg.get("say_clones", False))

    def _say_voices(self, kind: str) -> list[_V]:
        kind = _kind(kind)
        if kind != "Быстрый" and not self._say_clones():
            kind = _("Быстрый")
        out = [v for v in self._voices if v.ready == (kind == "Быстрый")]
        if kind == "Быстрый":
            # не скачанные голоса каталога - тоже в списке, с пометкой «скачать» (3.7)
            from core import voice_store as VS
            have, hidden = {v.id for v in out}, self._hidden_voices()
            for c in VS.CATALOG:
                if c.engine != "silero" and c.id not in have and c.id not in hidden:
                    x = _V(c.id, self._vname(c.id, c.title), c.language, True, c.engine, c.gender)
                    x.missing = True
                    out.append(x)
        return out

    @staticmethod
    def _say_label(v: _V) -> str:
        """«Евгений · муж. · русский» - имя, пол, язык (3.6)."""
        name = _(v.title).split(" (")[0]
        g = _(GENDER.get(v.gender, "")) if v.ready else _("клон")
        tail = _("скачать") if getattr(v, "missing", False) else ""
        return " · ".join(x for x in (name, g, _(LANG_MARK.get(v.language, v.language)), tail) if x)

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
            if _kind(self.say_mode.get()) != "Быстрый":
                self.say_mode.set(_("Быстрый"))
        v = self._say_voice()
        if v is not None and not v.ready:
            self.say_start_box.pack(fill="x", before=self.say_sound_head)
        else:
            self.say_start_box.pack_forget()

    def _say_on_kind(self, kind: str, remember: bool = True):
        kind = _kind(kind)
        if remember:
            self.cfg["say_mode"] = kind
            save_config(self.cfg)
        vs = self._say_voices(kind)
        labels = [self._say_label(v) for v in vs] or [_("(нет голосов)")]
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
        pool = self._voices + [v for v in self._say_voices("Быстрый") if getattr(v, "missing", False)]
        return next((v for v in pool if self._say_label(v) == lab), None)

    def _say_on_voice(self, remember: bool = True):
        v = self._say_voice()
        if v is not None and getattr(v, "missing", False):
            # серый голос - предложить скачать, а пока вернуть прежний выбор
            from core import voice_store as VS
            c = VS.by_id(v.id)
            prev = self.cfg.get("say_fast_voice") or FAST_VOICE
            back = next((x for x in self._say_voices("Быстрый") if x.id == prev
                         and not getattr(x, "missing", False)), None)
            if back is not None:
                self.say_voice_var.set(self._say_label(back))
            if c is not None and remember:
                self._vm_download(c, mirror=self.say_note)
            v = self._say_voice()
        if v is not None and remember:
            self.cfg["say_fast_voice" if v.ready else "say_clone_voice"] = v.id
            save_config(self.cfg)
        self._say_fx_show()
        self._say_layout()
        self._say_forecast()

    def _say_fx(self) -> VoiceFX:
        """Звучание голоса - сохранённое для него на странице «Голоса» (одно место)."""
        v = self._say_voice()
        if v is None or getattr(v, "missing", False):
            return VoiceFX()
        return load_fx(default_voices_dir(), v.id)

    def _say_fx_show(self):
        """«как есть» или «темп ×1,10 · высота +1,0 полутона» - что сохранено для голоса."""
        fx, d = self._say_fx(), _dec()
        parts = []
        if abs(fx.speed - 1) >= 0.01:
            parts.append(_("темп") + f" ×{fx.speed:.2f}".replace(".", d))
        for k, title, unit in (("pitch", _("высота"), _(" полутона")), ("bass", _("бас"), _(" дБ")),
                               ("brightness", _("яркость"), _(" дБ"))):
            val = getattr(fx, k)
            if val:
                parts.append(f"{title} {val:+.1f}".replace(".", d) + unit)
        if fx.denoise:
            parts.append(_("шумоподавление"))
        self.say_fx_line.configure(text=" · ".join(parts) if parts else _("как есть"))

    def _open_voice_fx(self, v):
        """«Звучание голоса ›»: страница «Голоса» с этим голосом справа."""
        self.show("voices")
        if v is not None and not getattr(v, "missing", False):
            self.select_fx_voice(v.id)

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
                return other.engine, {}, other.id, lang, _('Текст {0} - читает {1}. ').format(_(LANG_MARK.get(lang, lang)), _(other.title))
            hint = _('Быстрого голоса на языке текста нет - читает клон. ')
            vid = self._clone_for(lang)
        else:
            vid = v.id if v is not None else self._clone_for(lang)
        qv = self._voice_obj(vid) if vid else None
        if qv is not None and qv.qwen:
            return "mlxaudio", self._prof_kwargs(profiles.PROFILES["qwen"]), vid, lang, hint
        kw = self._prof_kwargs()
        if lang != "ru":
            kw.pop("variant", None)
        return self._clone_engine_for(), kw, vid, lang, hint

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
            self.say_forecast.configure(text=hint + _("Нет голоса-клона - добавьте образец."))
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
            maybe = _(" · возможны паузы") if clone and sp.rtf > 1.0 else ""
        else:
            wait = S.wait_needed(0.0, rest)
            maybe = ""
        self.say_forecast.configure(text=(
            _('{0}Звука ~{1} · звук через ~{2} с{3}').format(hint, _fmt_hms(audio), max(1, round(wait)), maybe)))

    # --- чтение ---
    def _say_buttons(self):
        on = self._live_kind == "say" and self._live is not None
        paused = on and self._live.paused
        playing = on and not paused
        self.say_btn.configure(
            text=_("Пауза") if playing else _("Продолжить") if paused else _("Прочитать"),
            state="disabled" if self.running and not on else "normal")
        T.restyle(self.say_btn, "primary", "pause" if playing else "play")
        self.say_stop_btn.configure(state="normal" if on else "disabled")

    def _say_toggle(self, *_a):
        """▶ / ❚❚: читает - пауза; на паузе - дальше; иначе - прочитать."""
        if self._live_kind == "say" and self._live is not None and not self._live.paused:
            self._say_pause()
        else:
            self.do_say()

    def _on_space(self, event):
        """Пробел на странице «Прочитать вслух» - ▶ / ❚❚, если курсор не в поле ввода."""
        import tkinter as tk
        if self.page != "say" or isinstance(event.widget, (tk.Text, tk.Entry)):
            return None
        self._say_toggle()
        return "break"

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
            self.say_note.configure(text=_("Вставьте текст или нажмите «Из буфера»"))
            return
        engine, kw, vid, _lang, hint = self._say_plan()
        if not vid:
            self.say_note.configure(text=_("Нет голоса-клона - добавьте образец в «Голосах»"))
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
            self.say_note.configure(text=_("Пауза · ▶ - продолжить"))
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
            self.say_note.configure(text=_("Начну с выделенного предложения · ▶"))

    def _say_save(self):
        if not self._say_last:
            return
        path = filedialog.asksaveasfilename(defaultextension=".mp3",
                                            filetypes=[("mp3", "*.mp3")],
                                            initialfile=_("Прочитано вслух.mp3"))
        if not path:
            return
        audio, sr, chain = self._say_last
        import tempfile
        from core.assemble import write_wav
        raw = os.path.join(tempfile.mkdtemp(prefix="herald_say_"), "raw.wav")
        write_wav(audio, sr, raw)
        try:
            polish_wav_to_mp3(raw, path, fx=chain)
            self.say_note.configure(text=_('Сохранено: {0}').format(os.path.basename(path)))
        except Exception as e:
            messagebox.showerror(_("Ошибка"), str(e))

    def _profile_listen(self, key: str):
        """Настройки, «Послушать»: фраза выбранным голосом и этим движком."""
        kind = "prof:" + key
        if self._live_kind == kind:
            self._live_stop()
            return
        v = self._voice_obj()
        if v is None or v.ready:
            self.prof_note.configure(text=_("Движок слышно на клоне: выберите голос-клон на главной"))
            return
        if v.qwen:
            self.prof_note.configure(text=_("Голос Qwen читает только движок Qwen: выберите "
                                            "на главной клон или обученный голос"))
            return
        p = profiles.get(key)
        kw = self._prof_kwargs(p)
        if v.language != "ru":
            if "variant" in kw:
                self.prof_note.configure(text=_("У ESpeech только русский голос"))
                return
        text = self.FX_PHRASE.get(v.language, self.FX_PHRASE["ru"])
        self._speak_job(kind, text, self._clone_engine_for(p), kw, v.id, prepared=True,
                        note=self.prof_note)

    # --- озвучка для других программ ---
    def _srv_sync(self):
        """Голос и движок для сервиса - из окна (читаются в главном потоке)."""
        v = self._voice_obj()
        self._srv_voice = (v.id if v is not None and not v.ready and not v.qwen
                           else self._clone_for("ru"))
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
            self.srv_note.configure(text=_("Порт - число от 1024 до 65535"), text_color=T.WARN)
            return
        idle = int(self.srv_idle.get().split()[0])
        self.cfg.update(speak_port=port, speak_idle=idle)
        save_config(self.cfg)
        on = bool(self.srv_var.get())
        if self._srv is not None and (not on or self._srv.port != port):
            self._srv.stop()
            self._srv = None
        if not on:
            self.srv_note.configure(text=_("Выключен"), text_color=T.MUTED)
            return
        if self._srv is None:
            srv = SpeakServer(port=port, idle_min=idle,
                              default_voice=lambda: self._srv_voice,
                              profile_kwargs=lambda: dict(self._srv_kwargs),
                              log=lambda m: self._logln(m))
            try:
                srv.start()
            except OSError as e:
                self.srv_note.configure(text=_('Порт {0} занят ({1})').format(port, e.strerror),
                                        text_color=T.WARN)
                return
            self._srv = srv
        self._srv.idle_min = idle
        self.srv_note.configure(text=_('Работает: {0}/speak').format(self._srv.url), text_color=T.OK)

    def _fx_save(self):
        if not self._fx_voice:
            return
        fx = self._fx_current()
        # шумоподавление - своя настройка («Дополнительно»): «Сохранить» её не обнуляет
        fx.denoise = load_fx(default_voices_dir(), self._fx_voice).denoise
        save_fx(default_voices_dir(), self._fx_voice, fx)
        self.fx_note.configure(text=_('Сохранено для «{0}»').format(
            self._vname(self._fx_voice, self._fx_voice)))
        if hasattr(self, "say_fx_line"):
            self._say_fx_show()
        self._refresh_voice_rows()
        self.show_estimate()           # темп влияет на длину и прогноз

    def load_sample(self):
        p = filedialog.askopenfilename(
            title=_("Запись нужного диктора (лучше аудиокнига, связная проза)"),
            filetypes=[(_("Аудио"), "*.mp3 *.wav *.m4a *.m4b *.flac *.ogg"), (_("Все файлы"), "*.*")])
        if not p:
            return
        name = ctk.CTkInputDialog(title=_("Имя голоса"), text=_("Как назвать этот голос?")).get_input()
        if not name or not name.strip():
            return
        name = name.strip()
        old = next((v for v in self._all_voices() if v.id == name and not v.ready and not v.qwen), None)
        if old is not None:
            # тот же голос молча не перезаписываем (задание 12): прежний - в корзину
            if not messagebox.askyesno(_("Голос уже есть"), _('Голос «{0}» уже есть. Заменить? '
                                                             '(прежний - в корзину)').format(name)):
                return
            self._voice_trash(old.id, ask=False)
        # ход - здесь, на странице «Голоса»; книгу в это время не начать (видеокарта одна)
        self.running = True
        self.add_voice_btn.configure(state="disabled", text=_("Готовлю образец…"))
        self.start_btn.configure(state="disabled")
        self._vm_note_undo(_("Слушаю запись и выбираю отрывок для образца…"), None)

        def end(vid=None, err=None):
            self.running = False
            self.add_voice_btn.configure(state="normal", text=_("Добавить образец…"))
            self._start_style()
            self._live_buttons()
            if err is not None:
                self.vm_note.configure(text=_("Не удалось подготовить голос") + ": " + tr_msg(str(err)))
                messagebox.showerror(_("Не удалось подготовить голос"), str(err))
                return
            self.vm_note.configure(text=_('Голос готов: {0}').format(vid))
            self.cfg.update(voice=vid)
            self.refresh_voices()
            self._refresh_voice_rows()
            self.select_fx_voice(vid)

        def work():
            try:
                vid = get_engine(clone_engine()).prepare_reference(p, name=name)
                self.ui(lambda: end(vid))
            except Exception as e:
                log_problem("не удалось подготовить голос", e, show=False)
                self.ui(lambda: end(err=e))
        threading.Thread(target=work, daemon=True).start()

    # ================= книга =================
    def on_drop(self, event):
        paths = [p for p in self.tk.splitlist(event.data) if p.lower().endswith(BOOK_EXT)]
        if not paths:
            messagebox.showwarning(_("Не та книга"), _("Поддерживаются: ") + ", ".join(BOOK_EXT))
            return
        if self.running:
            return self._busy_book_msg()
        self._set_book(paths[0])

    def choose_book(self):
        if self.running:
            return self._busy_book_msg()
        p = filedialog.askopenfilename(
            title=_("Книга"), filetypes=[(_("Книги"), " ".join("*" + e for e in BOOK_EXT)),
                                      (_("Все файлы"), "*.*")])
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
        self.status.configure(text=_("Перетащите книгу - и нажмите «Озвучить книгу»."))
        self.timer.configure(text=_fmt_clock(0))
        self.log.delete("1.0", "end")
        self._show_resume(None, None)
        self._update_crumb()
        self._recent_show()
        self._start_style()

    def _set_book(self, p: str):
        self.book_path = p
        self.empty.pack_forget()
        self.info.pack(fill="both", expand=True)
        self.book_title.configure(text=os.path.splitext(os.path.basename(p))[0])
        self.book_author.configure(text=os.path.dirname(p))
        self._clear_parts()
        self._update_crumb()
        self._recent_add(p)
        self._start_style()
        self._service_line()
        self.reestimate()

    # --- служебные файлы книги (задание 12) ---
    def _service_line(self):
        if not self.book_path:
            self.svc_line.grid_forget()
            return
        path = self.book_path

        def work():
            try:
                n = service_size(path)
            except Exception:
                n = 0
            self.ui(lambda: self._service_show(path, n))
        threading.Thread(target=work, daemon=True).start()

    def _service_show(self, path, n: int):
        if path != self.book_path:
            return
        if not n:
            self.svc_line.grid_forget()
            return
        self.svc_lbl.configure(text=_('Служебные файлы: {0}').format(_mb(n)))
        self.svc_clear.configure(state="disabled" if self.running else "normal",
                                 text=_("после озвучки") if self.running else _("Очистить"))
        self.svc_line.grid(row=3, column=1, sticky="w", pady=(4, 0))

    def _service_open(self):
        d = os.path.join(output_dir_for(self.book_path), SERVICE_DIR)
        _open_path(d if os.path.isdir(d) else output_dir_for(self.book_path))

    def _service_clear(self):
        if self.running or not self.book_path:
            return
        msg = _("Служебные файлы книги - в корзину (30 дней)?")
        if self._resume:
            msg = _("Книга не досчитана: без служебных файлов «Продолжить» начнёт её заново. "
                    "Очистить? (в корзину на 30 дней)")
        if not messagebox.askyesno(_("Очистить"), msg):
            return
        try:
            trash_service(self.book_path)
        except Exception as e:
            log_problem("служебные файлы не убрались", e)
        self._service_line()
        self._check_resume()
        if hasattr(self, "trash_rows"):
            self._refresh_trash()

    def single_file(self) -> bool:
        return self._len() == SINGLE_FILE

    def file_minutes(self) -> float:
        if self.single_file():
            return SINGLE_PART_MINUTES
        return float(self._len().split()[0])

    def reestimate(self):
        if not self.book_path:
            return
        self.est = None
        self.est_label.configure(text=_("Считаю объём книги…"))
        path, minutes = self.book_path, self.file_minutes()

        def work():
            try:
                est = estimate_book(path, minutes)
                self.ui(lambda: self._got_estimate(path, est))
            except Exception as e:
                log_problem("книга не прочиталась", e, show=False)
                self.ui(lambda: self.est_label.configure(text=_('Не удалось прочитать книгу: {0}').format(e)))
        threading.Thread(target=work, daemon=True).start()

    def _got_estimate(self, path, est):
        if path != self.book_path:
            return
        self.est = est
        from core.chapters import clean_book_title
        self.book_title.configure(text=clean_book_title(est.title) if est.title
                                  else os.path.basename(path))
        self.book_author.configure(text=est.author or _("автор не указан"))
        self._update_crumb()
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
        kbps = FORMAT_KBPS.get(self._fmt(), 192)
        mb = audio_sec * kbps / 8 / 1024
        size = (_('{0:.1f} ГБ').format(mb / 1024).replace(".", _dec()) if mb >= 1024
                else _('{0:.0f} МБ').format(mb))
        lang = {"ru": _("русская книга"), "en": _("английская книга")}.get(getattr(e, "language", "ru"), "")
        parts = _("один файл") if self.single_file() else _files(e.chapters)
        wait = audio_sec * self._rtf()
        # 3.8: куски, уже посчитанные этим голосом и режимом, берутся из кэша за
        # секунды - ждать только остальные («Проба пера 3» второй раз: 2:19 против 0:55)
        ci = self._cache_info_for()
        done_line = ""
        if ci and ci["cached"]:
            left = 1.0 - ci["cached_chars"] / max(1, ci["chars"])
            wait = wait * left + CACHED_CHUNK_SEC * ci["cached"]
            done_line = "\n" + _('{0} из {1} кусков уже посчитаны - они возьмутся из готового.').format(ci["cached"], ci["n"])
        night = _("  Удобно поставить на ночь.") if wait > 3 * 3600 else ""
        self.est_label.configure(text=(
            _('{0} · {1} · звука {2}\nЖдать около {3} · на диске ~{4}.{5}').format(lang, parts, _fmt_hms(audio_sec), _fmt_hms(wait), size, night) + done_line))
        self._check_resume()

    def _cache_info_for(self):
        """Что из книги уже в кэше синтеза (считается в фоне, 3.8). None - не знаем."""
        if not self.book_path or self.running:
            return None
        engine, kw = self._eng_choice()
        voice = self.current_voice()
        key = (self.book_path, engine, repr(sorted(kw.items())), voice,
               getattr(self.est, "language", "ru"))
        if getattr(self, "_cache_key", None) == key:
            return self._cache_info
        self._cache_key, self._cache_info = key, None
        lang = key[4]

        def work():
            info = None
            try:
                info = cached_share(self.book_path, engine=engine, eng_kwargs=kw,
                                    voice=voice, language=lang)
            except Exception as e:
                log_problem("не узнал, что из книги уже в кэше", e, show=False)

            def done():
                if getattr(self, "_cache_key", None) == key:
                    self._cache_info = info
                    if info and info["cached"]:
                        self.show_estimate()
            self.ui(done)
        threading.Thread(target=work, daemon=True).start()
        return None

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
        self.parts_title.configure(text=_('Готовые части · {0}').format(self._n_parts))

    def _load_existing_parts(self):
        """Части, оставшиеся с прошлого раза, - сразу в список."""
        self._clear_parts()
        try:
            d = output_dir_for(self.book_path)
        except Exception:
            return
        exts = tuple(v[0] for v in FORMATS.values())
        base = os.path.splitext(self.book_path)[0] + _(" - пробник")
        for ext in sorted(set(exts)):                 # пробник лежит рядом с книгой (3.2.8)
            if os.path.isfile(base + ext):
                self._add_part(_("Пробник"), base + ext, _("с прошлого раза"))
        if not os.path.isdir(d):
            return
        for f in sorted(os.listdir(d)):
            if f.startswith("Часть ") and f.endswith(exts) or (
                    f.endswith(exts) and not f.startswith(".")):
                self._add_part(os.path.splitext(f)[0], os.path.join(d, f), _("с прошлого раза"))

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
            self.start_btn.configure(text=_('Продолжить ({0} из {1})').format(info['done'], info['total']))
            if not self.restart_btn.winfo_manager():
                self.restart_btn.pack(side="right", padx=(0, 8), after=self.start_btn)
        else:
            self.start_btn.configure(text=_("Озвучить книгу"))
            self.restart_btn.pack_forget()

    def do_restart(self):
        if self.running or not self.book_path:
            return
        info = self._resume or {}
        if not messagebox.askyesno(_("Начать заново"), _('Стереть готовые части ({0} из {1}) и начать '
                                                       'книгу сначала?').format(info.get("done", 0), info.get("total", 0))):
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
        if v is not None and v.qwen:          # голос Qwen - всегда своим движком (3.3.2)
            return "mlxaudio", self._prof_kwargs(profiles.PROFILES["qwen"])
        kw = self._prof_kwargs()
        if v is not None and v.language != "ru":
            kw.pop("variant", None)
        return self._clone_engine_for(), kw

    def _clone_engine_for(self, p: profiles.Profile | None = None) -> str:
        """Движок клона по профилю: Qwen ru-stress - свой (mlx-audio, 3.8 / 10а)."""
        p = p or self.profile()
        return "mlxaudio" if p.engine == "mlxaudio" else clone_engine()

    # совместимость с тестами и CLI
    def _eng_kwargs(self) -> dict:
        return self._eng_choice()[1]

    def _conv_options(self) -> ConvertOptions:
        pauses = {k: self._pause(k) for k in self._pause_vars}
        return ConvertOptions(make_m4b=False, make_chapter_mp3=True,
                              minutes_per_file=self.file_minutes(),
                              audio_format=self._fmt(), single_file=self.single_file(),
                              strict_words=bool(self.strict_var.get()),
                              sound_check=bool(self.sound_var.get()),
                              cleanup_service=bool(self.cfg.get("cleanup_service", True)),
                              narrator=self._narrator(), **pauses)

    def _narrator(self) -> str:
        """Чтец для тегов файла: своё имя голоса («✎») или исходное, без «(муж.)»."""
        v = self._voice_obj()
        return (v.title if v else self.current_voice()).split(" (")[0]

    def _ready(self) -> bool:
        if not self.book_path or not os.path.isfile(self.book_path):
            messagebox.showwarning(_("Нет книги"), _("Перетащите книгу в окно."))
            return False
        if not self.current_voice():
            messagebox.showwarning(_("Нет голоса"), _("Добавьте голос на странице «Голоса»."))
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
        if hasattr(self, "cleanup_chk"):
            self.cleanup_chk.configure(state=st)
        if busy and self._live_kind != "preview":
            self._live_stop()                 # книга важнее прослушивания
        self._live_buttons()                  # «Прочитать», «Прослушать» - не во время счёта
        self._pause_flag.clear()
        self.pause_btn.configure(text=_("Пауза"), state="normal")
        T.restyle(self.pause_btn, "default", "pause")
        self._pause_visible(busy)
        if busy:
            self.start_btn.configure(text=_("Остановить"))
            T.restyle(self.start_btn, "danger", "square")
            self.restart_btn.pack_forget()
            self._stop_flag.clear()
            self._t0 = time.time()
            self._live_rtf = None
            self._synth_t0 = None
            self._prog_hist = []
            self._rate_base = self._rate_recent = None
            self._wait_lock = False
            self._step = 0
            self._tick()
        else:
            self.start_btn.configure(text=_("Озвучить книгу"), state="normal")
            T.restyle(self.start_btn, "primary", "play")
            self._start_style()
            if self.book_path:
                self._service_line()
            self.part_label.configure(text="")
            self.part_progress.set(0)
            if hasattr(self, "busy_line"):
                self.busy_line.pack_forget()
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
            self._tick_n = getattr(self, "_tick_n", 0) + 1
            if self._tick_n % 10 == 0 and not self._pause_flag.is_set():
                self._eta_refresh()                  # «осталось ≈» - каждые 10 с (3.7)
            self.after(1000, self._tick)

    # --- честный прогноз во время работы (3.7) ---
    def _eta(self) -> float | None:
        """Сколько ещё ждать - по скорости последних минут, а не всей работы:
        если видеокарту заняла другая программа, прогноз это видит сразу."""
        h = [x for x in self._prog_hist if x[1] < 1.0]
        if len(h) < 2:
            return None
        now = time.time()
        recent = [x for x in h if now - x[0] <= 180] or h[-2:]
        if len(recent) < 2:
            recent = h[-2:]
        (t0, f0), (t1, f1) = recent[0], recent[-1]
        if f1 - f0 < 0.005 or t1 - t0 < 1:
            return None
        rate = (f1 - f0) / (t1 - t0)
        self._rate_recent = rate
        if getattr(self, "_rate_base", None) is None and f1 >= 0.15:
            self._rate_base = (f1 - h[0][1]) / max(1.0, t1 - h[0][0])
        return max(0.0, (1.0 - f1) / rate - (now - t1))

    def _eta_refresh(self):
        if self._step != 2 or not self._prog_hist:
            return
        rest = self._eta()
        t = self.status.cget("text")
        if rest is not None and _(" · осталось ≈ ") in t:
            head, tail = t.split(_(" · осталось ≈ "), 1)
            tail = tail.split(" · ", 1)
            self.status.configure(text=_('{0} · осталось ≈ {1}').format(head, _fmt_hms(rest))
                                  + (" · " + tail[1] if len(tail) > 1 else ""))
        self._busy_check(rest)

    def _busy_check(self, rest):
        """Видеокарта занята другой программой - сказать и предложить подождать."""
        from core import busy
        owner = busy.lock_owner()
        slow = busy.slowed(getattr(self, "_rate_recent", None), getattr(self, "_rate_base", None))
        if owner or slow:
            who = _('программой «{0}»').format(owner) if owner else _("другой программой")
            eta = f" (≈ {_fmt_hms(rest)})" if rest is not None else ""
            self.busy_note.configure(text=_('Видеокарта занята {0} - озвучка идёт медленнее{1}').format(who, eta))
            if not self.busy_line.winfo_ismapped():
                self.busy_line.pack(fill="x", pady=(4, 0))
        elif self.busy_line.winfo_ismapped() and not self._wait_lock:
            self.busy_line.pack_forget()

    def _wait_free(self):
        """«Подождать, пока освободится»: пауза до снятия чужого замка."""
        if not self.running or self._wait_lock:
            return
        self._wait_lock = True
        if not self._pause_flag.is_set():
            self.do_pause()
        self.busy_btn.configure(state="disabled", text=_("Жду…"))
        self.status.configure(text=_("Пауза - жду, пока видеокарта освободится"))

        def poll():
            from core import busy
            if not self.running:
                self._wait_lock = False
                return
            if busy.lock_free():
                self._wait_lock = False
                # скорость до ожидания - уже не замер: иначе строка «занята» вернулась бы
                # по старой медленной скорости, хотя видеокарта свободна
                self._prog_hist = []
                self._rate_base = self._rate_recent = None
                self.busy_btn.configure(state="normal", text=_("Подождать, пока освободится"))
                self.busy_line.pack_forget()
                if self._pause_flag.is_set():
                    self.do_pause()                  # продолжить
                return
            self.after(10000, poll)
        self.after(10000, poll)

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
            step, what = 3, _("Сборка файлов")
        elif msg.startswith(("Озвучиваю", "Часть")):
            step, what = 2, _("Озвучка")
        else:
            step, what = 1, _("Подготовка текста: числа и ударения")
        self._step = step
        detail = msg
        left = ""
        if step == 2:
            sfr = max(0.0, min(1.0, (fr - 0.05) / 0.9))
            if self._synth_t0 is None:
                self._synth_t0, self._synth_fr = time.time(), sfr
            self._prog_hist.append((time.time(), sfr))
            rest = self._eta()
            if rest is not None:
                left = _(' · осталось ≈ {0}').format(_fmt_hms(rest))
            detail = tr_msg(msg.replace("Озвучиваю: ", ""))
            rtf = (f" · RTF {self._live_rtf:.2f}".replace(".", _dec())
                   if self._live_rtf else "")
            text = _('Шаг 2 из 3 · Озвучка - {0}, {1} %{2}{3}').format(detail, int(fr * 100), left, rtf)
        elif step == 3:
            text = _('Шаг 3 из 3 · {0}').format(what) if fr < 1 else _("Готово")
        elif msg.startswith("Скачиваю модели"):
            text = _('Шаг 1 из 3 · {0}').format(tr_msg(msg))      # первый запуск: честно, что идёт скачивание
        else:
            text = _('Шаг 1 из 3 · {0}').format(what)
        self.status.configure(text=text)

    def _part_progress(self, k: int, m: int, fr: float):
        text = _('Часть {0} из {1} - {2}%').format(k, m, int(fr * 100))
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
                self.pause_btn.configure(text=_("Пауза"))
                T.restyle(self.pause_btn, "default", "pause")
            else:
                live.pause()
                self.pause_btn.configure(text=_("Продолжить"))
                T.restyle(self.pause_btn, "primary", "play")
            return
        if self._pause_flag.is_set():                    # продолжить
            self._t0 += time.time() - self._paused_at    # пауза не в счёт
            if self._synth_t0 is not None:
                self._synth_t0 += time.time() - self._paused_at
            self._pause_flag.clear()
            keepawake.resume()
            self.pause_btn.configure(text=_("Пауза"))
            T.restyle(self.pause_btn, "default", "pause")
            self.status.configure(text=_("Продолжаю…"))
        else:
            self._paused_at = time.time()
            self._pause_flag.set()
            keepawake.pause()             # на паузе Mac может спать (пауза «на ночь»)
            self.pause_btn.configure(text=_("Продолжить"))
            T.restyle(self.pause_btn, "primary", "play")
            self.status.configure(text=_("Пауза - доделываю текущий кусок и жду. Готовое на месте"))

    def do_stop(self):
        if self._live_kind == "preview" and self._live is not None:
            self._live.stop()                 # звук - сразу, счёт - на ближайшем стыке
        self._stop_flag.set()
        self._pause_flag.clear()          # иначе стоящий на паузе синтез не узнает
        self.start_btn.configure(state="disabled")
        self.status.configure(text=_("Останавливаю после текущего куска…"))

    def _logln(self, text: str):
        self.ui(lambda: (self.log.insert("end", text + "\n"), self.log.see("end")))

    def _log_settings(self, what: str, engine: str, kw: dict, voice: str):
        fx = load_fx(default_voices_dir(), voice)
        e = self.est
        self.log.insert("end", f"{what}: {os.path.basename(self.book_path)}\n")
        if e is not None:
            self.log.insert("end", _('  книга: {0}, ~{1} звука, язык {2}\n').format(_files(e.chapters), _fmt_hms(e.audio_sec), getattr(e, 'language', '?')))
        size = _("одним файлом") if self.single_file() else _('части по {0}').format(self._len())
        self.log.insert("end", _('  голос: {0} · движок «{1}» ({2}, {3})\n  запись: {4} · {5}\n  звучание: темп ×{6:.2f}, высота {7:+.1f}, бас {8:+.1f}, яркость {9:+.1f}\n\n').format(voice, self.profile().title, engine, kw, self._fmt(), size, fx.speed, fx.pitch, fx.bass, fx.brightness))

    def do_preview(self):
        """«Проверить голос (1 мин)»: звук идёт по мере готовности, а в конце пробник
        ложится файлом рядом с книгой - как раньше."""
        if self.running or not self._ready():
            return
        path, voice = self.book_path, self.current_voice()
        engine, kw = self._eng_choice()
        opts, fmt = self._conv_options(), self._fmt()
        self._live_stop()
        self._set_busy(True, _("Шаг 1 из 3 · Готовлю пробник…"))
        live = self._live_new("preview", engine, kw, voice, opts.sound_check and _clone_engine(engine),
                              lambda t: self.status.configure(text=_("Пробник · ") + t))
        self.pause_btn.configure(state="normal")
        self.progress.set(0)
        self._log_settings(_("Пробник (одна минута)"), engine, kw, voice)
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
                out_path = os.path.splitext(path)[0] + _(" - пробник") + ext
                # с поправками голоса - как в книге (до 3.4 пробник шёл без них)
                vfx = load_fx(default_voices_dir(), voice)
                chain = (vfx.filter_chain() if _clone_engine(engine)
                         else S.with_tempo(vfx.filter_chain(), vfx.speed or 1.0))
                polish_wav_to_mp3(out, out_path, fmt=fmt, fx=chain)
                shutil.rmtree(os.path.dirname(out), ignore_errors=True)
                dur = len(a) / sr
                fs = live.first_sound
                self._logln(f"Пробник: {dur:.0f} с звука, счёт {_fmt_hms(time.time() - t0)}"
                            + (f", первый звук через {fs:.1f} с, запинок {live.stalls}"
                               if fs is not None else ""))
                self.ui(lambda: self._add_part(_("Пробник"), out_path, _('{0:.0f} с звука').format(dur)))
                if live.sink is None:
                    self._play_plain(a.astype("float32"), sr, "")
                live.wait()                    # «Остановить» обрывает и звук
                self.ui(lambda: self._preview_done(live, _("Пробник готов - он в списке, рядом с книгой")))
            except Cancelled:
                self.ui(lambda: self._preview_done(live, _("Остановлено")))
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
        self._set_busy(True, _("Шаг 1 из 3 · Подготовка текста: числа и ударения"))
        self.pause_btn.configure(state="normal")
        self.progress.set(0)
        self.log.delete("1.0", "end")
        self._log_settings(_("Озвучка книги"), engine, kw, voice)
        if not resuming:
            self._clear_parts()
        else:
            self._clear_parts()           # части с прошлого раза придут заново, по порядку
        t0 = time.time()

        def on_chapter(ci, title, mp3):
            name = os.path.splitext(os.path.basename(mp3))[0] if mp3 else _('Часть {0}').format(ci)
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
                what = _("один файл") if res.chapters == 1 else _files(res.chapters)
                self._logln(f"\nГотово. Звука {_fmt_hms(res.duration)}, счёт {_fmt_hms(took)} "
                            f"(на секунду звука {took / max(res.duration, 1):.2f} с)")
                self.ui(lambda: self._learn_book(res.duration, took, res.fresh_sec))
                if res.mp3_dir:
                    self._logln(f"Папка: {res.mp3_dir}")

                def done():
                    tail = _(" · служебные файлы - в корзине") if res.service_trashed else ""
                    self._set_busy(False, _('Готово: {0}, звука {1}').format(what, _fmt_hms(res.duration)) + tail)
                    if res.service_trashed and hasattr(self, "trash_rows"):
                        self._refresh_trash()
                    if opts.single_file and res.mp3_dir:
                        self._load_existing_parts()
                    self._check_resume()
                self.ui(done)
            except Cancelled:
                self._logln("Остановлено. Готовые части остались на диске.")
                self.ui(lambda: (self._set_busy(False, _("Остановлено - готовое на месте")),
                                 self._check_resume()))
            except ScannedPdfError as e:
                self._fail(e, _("PDF без текстового слоя (скан)"))
            except ReaderError as e:
                self._fail(e, _("Не удалось прочитать книгу"))
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

    def _fail(self, e, title=_("Ошибка")):
        log_problem(title, e, show=False)
        self.ui(lambda: self._details_visible(True))
        self.ui(lambda: (self._set_busy(False, _("Ошибка - подробности в «Подробностях»")),
                         messagebox.showerror(title, str(e))))

    # ================= журнал =================
    def _enable_copy(self, box):
        """Cmd+C и правая кнопка в журнале - привязки на внутреннем виджете."""
        inner = getattr(box, "_textbox", box)
        import tkinter as tk
        menu = tk.Menu(inner, tearoff=0)
        menu.add_command(label=_("Скопировать выделенное"), command=lambda: self.copy_log(selection=True))
        menu.add_command(label=_("Скопировать всё"), command=lambda: self.copy_log())
        menu.add_separator()
        menu.add_command(label=_("Выделить всё"), command=lambda: inner.tag_add("sel", "1.0", "end"))

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
        self.copy_btn.configure(text=_("Скопировано"))
        self.after(1600, lambda: self.copy_btn.configure(text=_("Копировать")))
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
    global _RESTART
    if not cfg.get("ui_lang"):
        # по умолчанию - язык системы; кто пользовался Herald до 3.7 (окно было
        # только русским) - остаётся на русском, чтобы обновление не сменило язык
        cfg["ui_lang"] = "ru" if cfg else i18n.system_lang()
        save_config(cfg)
    i18n.set_lang(cfg["ui_lang"])
    ctk.set_widget_scaling(float(cfg.get("scale", DEFAULT_SCALE)))
    real_err = sys.stderr
    while True:
        _RESTART = None
        app = App()
        sys.stderr = App._StderrToLog(app, real_err)
        state = getattr(main, "_state", None)
        if state:                                   # после смены языка / меню
            if state.get("book") and os.path.isfile(state["book"]):
                app._set_book(state["book"])
            app.show(state.get("page") or "book")
        app.mainloop()
        if _RESTART is None:
            break
        main._state = _RESTART
        try:
            app.destroy()
        except Exception:
            pass
    # Выход без разбора объектов Metal (MLX) и onnxruntime: изредка он падал с
    # «recursive_mutex lock failed» (опыт Scribe 2.4)
    os._exit(0)


if __name__ == "__main__":
    main()
