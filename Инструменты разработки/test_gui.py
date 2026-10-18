#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон окна без человека: открыть, потыкать, поймать ошибки (окно 3.1).

Родился из истории, где одну и ту же кнопку ломали трижды подряд, а замечал
это только пользователь. Тест поднимает настоящее окно, дёргает обработчики и
ловит всё, что упало, - включая исключения из колбэков tkinter, которые иначе
просто печатаются в консоль и никого не останавливают.

    python "Инструменты разработки/test_gui.py"

Синтез здесь НЕ запускается: проверяется интерфейс, а не звук.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import os
import sys
import time
import traceback

os.environ.setdefault("PYTHONHASHSEED", "0")

FAILS: list[str] = []


def check(name: str, fn, limit: int = 30):
    import signal
    try:
        signal.alarm(limit)
        fn()
        signal.alarm(0)
        print(f"  OK    {name}")
    except BaseException as e:
        signal.alarm(0)
        FAILS.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  ПАДАЕТ {name}: {type(e).__name__}: {e}")


def must(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main() -> int:
    import tkinter
    import customtkinter as ctk
    import theme as T
    import tts_gui as G
    from core import pipeline as P
    from core.voicefx import load as load_fx, save as save_fx
    from core.engines.f5 import default_voices_dir

    def report(self, exc, val, tb):
        FAILS.append(f"колбэк tkinter: {val!r}")
        print(f"  ПАДАЕТ колбэк tkinter: {val!r}")
        traceback.print_exception(exc, val, tb)
    tkinter.Tk.report_callback_exception = report

    print("окно:")
    T.apply()
    cfg_backup = G.load_config()
    app = G.App()
    app.update()

    import signal

    def timeout(_sig, _frm):
        raise TimeoutError("шаг теста не уложился в отведённое время")
    signal.signal(signal.SIGALRM, timeout)

    def wait(cond, seconds=10.0):
        """Крутить НАСТОЯЩИЙ цикл событий, пока условие не выполнится."""
        end = time.time() + seconds

        def tick():
            if cond() or time.time() > end:
                app.quit()
            else:
                app.after(50, tick)
        app.after(50, tick)
        app.mainloop()
        return cond()

    check("открылось и отрисовалось", lambda: app.update_idletasks())
    check("меню «Правка» на месте", lambda: app.nametowidget(app.cget("menu")))
    check("страницы построены заранее (3.5: «Прочитать вслух», 3.7: «Справка»)",
          lambda: must(set(app.pages) == {"book", "say", "voices", "help", "settings"}, app.pages.keys()))

    print("боковое меню и темы:")

    def pages_switch():
        for k in ("voices", "say", "help", "settings", "book"):
            app.show(k)
            app.update()
            must(app.page == k and app.pages[k].winfo_ismapped(), f"страница {k} не видна")
            others = [o for o in app.pages if o != k and app.pages[o].winfo_ismapped()]
            must(not others, f"видны лишние страницы: {others}")
    check("страницы переключаются, видна ровно одна", pages_switch)

    def themes():
        for mode in ("light", "dark", "system"):
            app.set_theme(mode)
            app.update()
            must(ctk.get_appearance_mode().lower() in ("light", "dark"), "тема не встала")
        app.set_theme(cfg_backup.get("theme", "system"))
    check("светлая, тёмная и системная тема", themes)
    check("кнопка темы перебирает режимы",
          lambda: (app._cycle_theme(), app._cycle_theme(), app._cycle_theme(),
                   must(app.cfg["theme"] == cfg_backup.get("theme", "system"), app.cfg["theme"])))

    print("голоса:")
    check("список голосов заполнен",
          lambda: must(app.voice_menu.cget("values") != [""], "пусто"))
    check("есть клоны и готовые голоса",
          lambda: must(any(v.ready for v in app._all_voices())
                       and any(not v.ready for v in app._all_voices()), "нет одной из групп"))

    def voices_page():
        app.show("voices")
        app.update()
        n = len(app.voice_rows.winfo_children())
        must(n >= 4, f"строк голосов: {n}")
    check("страница «Голоса» показывает обе группы", voices_page)

    vid = next(v.id for v in app._all_voices() if not v.ready)
    before = load_fx(default_voices_dir(), vid)

    def no_sound_button():
        texts = []

        def walk(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkButton):
                    texts.append(c.cget("text"))
                walk(c)
        walk(app.voice_rows)
        must("Звучание" not in texts, f"кнопки в списке голосов: {texts}")
    check("кнопки «Звучание» больше нет", no_sound_button)

    def row_click():
        app.show("voices")
        app.update()
        row = app._voice_row_widgets[vid]
        # настоящий щелчок мышью - по подписи с именем внутри строки
        name = [c for c in row.winfo_children() if isinstance(c, ctk.CTkLabel)][1]
        inner = getattr(name, "_label", name)
        inner.event_generate("<Button-1>", x=2, y=2)
        app.update()
        must(app._fx_voice == vid and app.fx_title.cget("text"), "настройки не открылись")
        must(app.current_voice() == vid, f"голос не выбран: {app.current_voice()}")
        must(str(app.fx_play_btn.cget("state")) == "normal", "«Прослушать» выключена")
        row = app._voice_row_widgets[vid]
        must(row.cget("fg_color") == T.ACCENT_SOFT, "выбранная строка не подсвечена")
        other = next(k for k in app._voice_row_widgets if k != vid)
        orow = app._voice_row_widgets[other]
        app._voice_row_widgets[other] = orow
        orow.configure(fg_color=app._voice_row_color(other, True))
        must(orow.cget("fg_color") == T.SURFACE2, "наведение не подсвечивает")
        must(app._voice_row_color(other, False) == "transparent", "подсветка не гаснет")
    check("строка голоса: щелчок выбирает и открывает настройки, наведение подсвечивает",
          row_click)

    def fx_save():
        sl, _l, _u = app._fx_vars["bass"]
        sl.set(2.5)
        app._fx_changed("bass")
        app._fx_save()
        must(abs(load_fx(default_voices_dir(), vid).bass - 2.5) < 0.01, "не сохранилось")
        save_fx(default_voices_dir(), vid, before)       # вернуть как было
    check("поправка звучания сохраняется и читается", fx_save)
    app.show("book")

    print("книга:")
    book = "Примеры текстов/Пример - проба пера.txt"
    check("выбор книги", lambda: app._set_book(book))
    wait(lambda: app.est is not None)
    check("оценка посчиталась", lambda: must(app.est is not None, "оценки нет"))
    check("в оценке честное время ожидания",
          lambda: must("Ждать около" in app.est_label.cget("text"), app.est_label.cget("text")))
    check("язык книги определён", lambda: must(app.est.language in ("ru", "en"), app.est.language))

    def resume_button():
        import json as _j
        sd = os.path.join(P.output_dir_for(book), P.STATE_DIR)
        os.makedirs(sd, exist_ok=True)
        engine, kw = app._eng_choice()
        fp = P.settings_fingerprint(engine, dict(kw, language=app.est.language),
                                    app.current_voice(), app._conv_options())
        open(os.path.join(sd, "part0001.flac"), "wb").close()
        _j.dump({"fingerprint": fp, "total": 3, "finished": False,
                 "parts": {"1": {"flac": "part0001.flac"}}},
                open(os.path.join(sd, P.STATE_FILE), "w"))
        try:
            app._check_resume()
            wait(lambda: "Продолжить" in app.start_btn.cget("text"))
            must("1 из 3" in app.start_btn.cget("text"), app.start_btn.cget("text"))
            must(app.restart_btn.winfo_manager(), "нет кнопки «Начать заново»")
            P.forget_progress(book)
            app._check_resume()
            wait(lambda: app.start_btn.cget("text") == "Озвучить книгу")
            must(not app.restart_btn.winfo_manager(), "«Начать заново» не спряталась")
        finally:
            P.forget_progress(book)
            try:
                os.rmdir(P.output_dir_for(book))
            except OSError:
                pass
    check("«Продолжить» при незаконченной книге", resume_button)

    print("движок:")

    def profiles_switch():
        from core import profiles
        f5 = [k for k, p in profiles.PROFILES.items() if p.engine.startswith("f5")]
        must(f5 == ["main", "espeech", "draft"], list(profiles.PROFILES))
        shown = [p for p in profiles.PROFILES.values() if profiles.available(p)]
        must(len(app._profile_radios) == len(shown), "в Настройках не те движки")
        clone = next(v for v in app._voices if not v.ready)
        app.voice_var.set(clone.label())
        # 3.8: Qwen ru-stress - свой движок (mlx-audio), виден только там, где может работать
        if profiles.available(profiles.PROFILES["qwen"]):
            app.profile_var.set("qwen")
            app._on_profile()
            must(app._eng_choice()[0] == "mlxaudio", f"Qwen: {app._eng_choice()}")
        for key in f5:
            app.profile_var.set(key)
            app._on_profile()
            app.update()
            engine, kw = app._eng_choice()
            must(engine == "f5mlx" and kw == profiles.engine_kwargs(profiles.get(key), app._quality()),
                 f"{key}: {engine} {kw}")
            must(kw.get("steps") != 32, "32 шага в окне")
        app.profile_var.set("main")
        app._on_profile()
        e, kw = app._eng_choice()
        must(kw.get("steps") == 12 and "variant" not in kw, f"основной: {kw}")
        # 3.6: по умолчанию «Сбалансировано» - подсказка на 6 шагах из 12
        must(app._quality() == "balanced" or G.load_config().get("quality"), app._quality())
        for q, k in (("fast", 4), ("balanced", 6), ("max", None)):
            app.quality_var.set(q)
            app._on_quality()
            must(app._eng_choice()[1].get("cfg_steps") == k, f"{q}: {app._eng_choice()[1]}")
            must(q in app._rtf_key() and G.VERSION in app._rtf_key(), app._rtf_key())
        app.quality_var.set(cfg_backup.get("quality", "balanced"))
        app._on_quality()
        ready = next(v for v in app._voices if v.ready)
        app.voice_var.set(ready.label())
        must(app._eng_choice()[0] == "silero", "готовый голос - не Silero")
        must(app._rtf() == profiles.SILERO.rtf, "прогноз для Silero")
        app.voice_var.set(clone.label())
        must(any(v.ready for v in app._voices) and any(not v.ready for v in app._voices),
             "в списке озвучки не все голоса")
    check("движки в Настройках (основной, ESpeech, черновик, Qwen там, где есть), Silero - выбором голоса",
          profiles_switch)
    app.voice_var.set(next(v.label() for v in app._voices if v.id == cfg_backup.get("voice"))
                      if any(v.id == cfg_backup.get("voice") for v in app._voices)
                      else app.voice_var.get())

    def accel():
        app._show_accel("cpu")
        t = app.accel_hint.cget("text")
        must("суток" in t and "Silero" in t, t)
        wait_cpu = app._rtf()
        app._show_accel("mlx")
        must("Apple" in app.accel_hint.cget("text"), app.accel_hint.cget("text"))
        # с прикидкой видеоядра, а не с замером: замер у Андрея мог быть снят
        # под чужой нагрузкой (3,25 вместо ~1,2) - тест не должен от этого зависеть
        from core import profiles
        mlx_prior = profiles.rtf_for(app.profile(), "mlx", app._quality())
        must(wait_cpu > mlx_prior * 5, f"прогноз на процессоре {wait_cpu} против {mlx_prior}")
    check("чем считается: на процессоре - предупреждение и честный прогноз", accel)

    print("паузы:")

    def pauses():
        o = app._conv_options()
        must((o.gap_sec, o.para_gap_sec, o.dialog_gap_sec) == (0.4, 0.8, 1.1) or
             cfg_backup.get("pauses"), f"по умолчанию: {o.gap_sec}, {o.para_gap_sec}, {o.dialog_gap_sec}")
        sl, _v = app._pause_vars["dialog_gap_sec"]
        sl.set(1.5)
        app._on_pause("dialog_gap_sec")
        must(abs(app._conv_options().dialog_gap_sec - 1.5) < 1e-6, "пауза не дошла до озвучки")
        must(abs(app.cfg["pauses"]["dialog_gap_sec"] - 1.5) < 1e-6, "пауза не запомнилась")
        app._pauses_reset()
        o = app._conv_options()
        must((round(o.gap_sec, 2), round(o.para_gap_sec, 2), round(o.dialog_gap_sec, 2))
             == (0.4, 0.8, 1.1), "«Как было» не вернула значения")
    check("три паузы в Настройках: по умолчанию 0,4 / 0,8 / 1,1, доходят до озвучки", pauses)

    def checks():
        """3.3: проверка по звуку включена по умолчанию; строгой 3.2 в окне нет."""
        must(app.sound_var.get() is True or cfg_backup.get("sound_check") is False,
             "проверка по звуку по умолчанию включена")
        must(not hasattr(app, "strict_chk"), "строгая проверка 3.2 убрана из окна")
        o = app._conv_options()
        must(o.sound_check == app.sound_var.get() and o.strict_words is False,
             "настройки не дошли до озвучки")
        app.sound_var.set(False); app._on_strict()
        cheap = app._rtf()
        must(app._conv_options().sound_check is False, "проверка по звуку не выключается")
        app.sound_var.set(True); app._on_strict()
        must(app._rtf() > cheap or app.cfg.get("rtf", {}).get(app._rtf_key()),
             "прогноз с проверкой по звуку дольше")
    check("проверка слов по звуку включена, строгой 3.2 в окне нет", checks)

    def rtf_live():
        app._set_busy(True, "тест")
        app._got_rtf(2.05)
        app._show_progress(0.3, "Озвучиваю: часть 1 из 2")
        t = app.status.cget("text")
        app._set_busy(False)
        must("RTF 2,05" in t, t)
        must(abs(app.cfg["rtf"][app._rtf_key()] - 2.05) < 0.5, app.cfg.get("rtf"))
        must("замерено" in app.tech_label.cget("text"), "замер не показан в Настройках")
    check("честный RTF: в строке состояния и в Настройках", rtf_live)
    app.profile_var.set(cfg_backup.get("profile", "main"))
    app._on_profile()

    print("подробности:")
    def details_hidden():
        app.show("book")
        app.update()
        must(not app.details_box.winfo_ismapped() or app.cfg.get("show_log"),
             "«Подробности» видны без ошибки")
        import io
        tee = G.App._StderrToLog(app, io.StringIO())   # тот же перехват, что в окне
        tee.write("<torch_package_0>.x.py:286: SyntaxWarning: invalid escape sequence '\\^'\n")
        tee.write("ℹ Продолжаю: готовых частей 1 из 2\n")
        for _ in range(30):
            app.update()
            time.sleep(0.03)
        must("SyntaxWarning" not in app.log.get("1.0", "end"), "предупреждение библиотеки в окне")
        must("Продолжаю" in app.log.get("1.0", "end"), "справка программы пропала")
        app._details_visible(True)                     # при ошибке - появляются
        app.update()
        must(app.details_box.winfo_ismapped(), "при ошибке «Подробности» не показаны")
    check("«Подробности» скрыты, предупреждения библиотек - не в окне", details_hidden)
    check("подробности разворачиваются", lambda: (app.toggle_details(), app.update(),
                                                  must(app.log.winfo_ismapped(), "не видно")))

    def copy():
        app.log.delete("1.0", "end")
        app.log.insert("end", "проверка копирования")
        app.copy_log()
        must(app.clipboard_get() == "проверка копирования", app.clipboard_get())
    check("«Копировать» кладёт журнал в буфер", copy)
    check("свернуть подробности", lambda: (app.toggle_details(), app.update(),
                                           must(not app.log.winfo_ismapped(), "видно")))
    inner = app.log._textbox
    check("Cmd+C на внутреннем виджете", lambda: must(inner.bind("<Command-c>"), "нет привязки"))

    print("работа:")

    def busy():
        app._set_busy(True, "тест")
        must(app.start_btn.cget("text") == "Остановить", app.start_btn.cget("text"))
        locked = [w for w in app._setting_widgets + [app.preview_btn, app.add_voice_btn]
                  if str(w.cget("state")) != "disabled"]
        must(not locked, f"не заблокировано: {len(locked)}")
        app._set_busy(False)
        must(app.start_btn.cget("text") == "Озвучить книгу", "кнопка не вернулась")
    check("во время работы кнопка - «Остановить», настройки заперты", busy)

    def pause_cycle():
        import threading
        app._set_busy(True, "тест")
        app.pause_btn.configure(state="normal")
        app.do_pause()
        must(app._pause_flag.is_set() and app.pause_btn.cget("text") == "Продолжить", "пауза")
        passed = threading.Event()
        th = threading.Thread(target=lambda: (app._pause_gate(), passed.set()))
        th.start()
        time.sleep(0.8)
        must(not passed.is_set(), "синтез не ждёт на паузе")
        app.do_pause()
        th.join(2)
        app._set_busy(False)
        must(passed.is_set(), "после «Продолжить» синтез не пошёл")
    check("«Пауза» и «Продолжить»", pause_cycle)

    def stop_breaks_pause():
        import threading
        app._set_busy(True, "тест")
        app.do_pause()
        th = threading.Thread(target=app._pause_gate)
        th.start()
        app.do_stop()
        th.join(2)
        alive = th.is_alive()
        app._set_busy(False)
        must(not alive, "«Стоп» не снимает паузу")
    check("«Остановить» работает и на паузе", stop_breaks_pause)

    def steps_words():
        app._set_busy(True, "тест")
        app._show_progress(0.02, "Готовлю текст (числа, ударения)…")
        must(app.status.cget("text").startswith("Шаг 1 из 3"), app.status.cget("text"))
        app._show_progress(0.2, "Озвучиваю: часть 4 из 27")
        app._synth_t0 -= 60
        app._prog_hist = [(t_ - 60, f_) for t_, f_ in app._prog_hist]   # минуту назад
        app._show_progress(0.3, "Озвучиваю: часть 5 из 27")
        t = app.status.cget("text")
        must(t.startswith("Шаг 2 из 3 · Озвучка - часть 5 из 27, 30 %") and "осталось" in t, t)
        app._show_progress(0.97, "Склеиваю части в один файл…")
        must(app.status.cget("text").startswith("Шаг 3 из 3"), app.status.cget("text"))
        app._set_busy(False)
    check("этапы словами и оставшееся время", steps_words)

    def part_progress():
        app._part_progress(2, 5, 0.5)
        wait(lambda: app.part_label.cget("text") == "Часть 2 из 5 - 50%", 3)
        must(app.part_label.cget("text") == "Часть 2 из 5 - 50%", app.part_label.cget("text"))
    check("прогресс текущей части", part_progress)

    def parts_survive_theme():
        """Тема сменилась, пока «Озвучка» скрыта: список частей не должен пропасть."""
        for i in range(3):
            app._add_part(f"Часть {i + 1}", os.path.abspath(book), "")
        app.show("settings")
        app.set_theme("dark" if ctk.get_appearance_mode() == "Light" else "light")
        app.show("book")
        wait(lambda: False, 0.5)
        rows = [w for w in app.rows.winfo_children() if isinstance(w, G.PartRow)]
        vis = sum(r.winfo_ismapped() for r in rows)
        app.set_theme(cfg_backup.get("theme", "system"))
        app._clear_parts()
        must(rows and vis == len(rows), f"видно {vis} из {len(rows)}")
    check("части видны после смены темы на другой странице", parts_survive_theme)

    def mini_player():
        import numpy as np
        import soundfile as sf
        import tempfile
        d = tempfile.mkdtemp(prefix="herald_fx_")
        files = []
        for i in range(2):
            f = os.path.join(d, f"часть{i}.wav")
            sf.write(f, np.zeros(24000 * 5, dtype=np.float32), 24000)
            files.append(f)
        app._add_part("Часть 1", files[0], "")
        app._add_part("Часть 2", files[1], "")
        app.update()
        rows = [w for w in app.rows.winfo_children() if isinstance(w, G.PartRow)]
        r1, r2 = rows[-2], rows[-1]
        app.listen(r1)
        app.update()
        must(r1.transport is not None and app.player.playing, "не заиграло")
        must(r1.time_lbl.cget("text").endswith("/ 0:05"), r1.time_lbl.cget("text"))
        app.player.toggle()
        r1.sync(app.player)
        must(not app.player.playing and r1._shown_playing is False, "пауза не встала")
        r1.seek.set(0.6)
        r1.seek._command(0.6)
        must(abs(app.player.position - 3.0) < 0.3, f"перемотка: {app.player.position}")
        must(r1.time_lbl.cget("text").startswith("0:03"), r1.time_lbl.cget("text"))
        app.player.toggle()
        must(app.player.playing, "играть после паузы")
        r1.stop_btn.invoke()
        must(not app.player.playing and app.player.position < 0.05, "стоп не вернул в начало")
        app.listen(r2)
        app.update()
        must(r1.transport is None and r2.transport is not None, "играет не одна часть")
        must(app.player.path == files[1] and app.player.playing, "вторая часть не играет")
        app.on_close.__func__                     # обработчик закрытия есть
        must(app.protocol("WM_DELETE_WINDOW"), "закрытие окна не перехвачено")
        app._close_player()
        must(not app.player.playing and r2.transport is None, "звук не остановился")
        app._clear_parts()
    check("мини-плеер: играть/пауза, стоп, время, перемотка, одна часть за раз", mini_player)

    def no_duplicate_folders():
        """3.2: у части одна кнопка папки; в шапке списка и в Настройках дублей нет."""
        def buttons(w):
            out = []
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkButton):
                    out.append(c)
                out += buttons(c)
            return out
        app._add_part("Часть 1", os.path.abspath(book), "")
        app.update()
        row = [w for w in app.rows.winfo_children() if isinstance(w, G.PartRow)][-1]
        others = [b for b in buttons(row) if b is not row.listen_btn]
        must(others == [row.reveal_btn], f"кроме «Слушать» у части кнопок: {len(others)}")
        must(row.reveal_btn.cget("text") == G.REVEAL_TEXT, row.reveal_btn.cget("text"))
        texts = [b.cget("text") for b in buttons(app.info)]
        must("Папка" not in texts, f"кнопка «Папка» над списком частей: {texts}")
        texts = [b.cget("text") for b in buttons(app.pages["settings"])]
        must("Папка голосов" not in texts, "дубль «Папка голосов» в Настройках")
        app._clear_parts()
    check("кнопки папки без дублей", no_duplicate_folders)

    def parts_list():
        n0 = app._n_parts
        app._add_part("Часть 1", os.path.abspath(book), "проверка")
        app.update()
        must(app._n_parts == n0 + 1, "строка не добавилась")
        app._clear_parts()
    check("список частей: строка со «Слушать» и Finder", parts_list)

    def clock():
        for sec, want in ((187, "3 мин 07 с"), (3912, "1 ч 05 мин 12 с")):
            must(G._fmt_clock(sec) == want, f"{sec} -> {G._fmt_clock(sec)}")
        must([G._files(n) for n in (1, 3, 5, 21)] == ["1 файл", "3 файла", "5 файлов", "21 файл"],
             "склонение")
    check("таймер и склонения", clock)

    def lengths():
        app.len_var.set("60 минут")
        o = app._conv_options()
        must(o.minutes_per_file == 60 and not o.single_file, "60 минут")
        app.len_var.set(G.SINGLE_FILE)
        o = app._conv_options()
        app.len_var.set("15 минут")
        must(o.single_file, "«Один файл» не передаётся")
    check("длина 60 минут и «Один файл»", lengths)
    check("очистка книги", lambda: (app.clear_book(), app.update(),
                                    must(app.empty.winfo_ismapped(), "зона не вернулась")))

    print("звук потоком (3.4):")
    import numpy as np
    from core import stream as S
    real_speak, real_sink, real_preview = G.S.speak, G.S.default_sink, G.preview_sample

    def fake_speak(text, *, engine, voice, live, eng_kwargs=None, prepared=False, **_k):
        """Заглушка синтеза: три куска по 0,8 с, каждый «считается» 0,3 с."""
        sr = 24000
        items = [("Первый кусок.", 0.4), ("Второй кусок.", 0.4), ("Третий.", 0.4)]
        if _k.get("on_spans"):
            n = max(1, len(text) // 3)
            _k["on_spans"]([(0, n), (n, 2 * n), (2 * n, len(text))])
        live.plan(items, sr)
        out = []
        for _t, _g in items:
            if live.should_stop():
                raise P.Cancelled()
            time.sleep(0.3)
            a = (0.1 * np.sin(np.arange(int(sr * 0.8)) / 5)).astype("float32")
            live.feed(a, 0.8)
            out.append(a)
        live.finish()
        return np.concatenate(out), sr

    def fake_preview(path, *, live=None, progress=None, **_k):
        a, sr = fake_speak("", engine="x", voice="v", live=live)
        import tempfile
        import soundfile as sf
        out = os.path.join(tempfile.mkdtemp(prefix="t2a_prev_"), "preview.wav")
        sf.write(out, np.concatenate([a] * 3), sr)
        return out

    G.S.speak, G.S.default_sink = fake_speak, lambda: S.ClockSink(realtime=True)
    G.preview_sample = fake_preview

    def pump(cond, limit=10.0):
        t = time.time()
        while time.time() - t < limit:
            app.update()
            if cond():
                return True
            time.sleep(0.03)
        return False

    def say_flow():
        app.show("say")
        app.say_box.delete("1.0", "end")
        app.say_box.insert("1.0", "Море к вечеру стихло. Над водой повис туман. Лодки вернулись.")
        app._say_text_changed()
        must("знаков" in app.say_count.cget("text"), "счётчик знаков")
        app.say_mode.set("Быстрый")
        app._say_on_kind("Быстрый")
        must("звук через" in app.say_forecast.cget("text"), app.say_forecast.cget("text"))
        app.do_say()
        live = app._live
        must(live is not None and app._live_kind == "say", "поток не начался")
        must(str(app.say_stop_btn.cget("state")) == "normal", "■ доступна во время чтения")
        must(pump(lambda: live.first_sound is not None), "звук не пошёл")
        must(pump(lambda: "Играет" in app.say_note.cget("text")), app.say_note.cget("text"))
        inner = app.say_box._textbox
        must(pump(lambda: bool(inner.tag_ranges("say_cur"))), "звучащее предложение не подсвечено")
        must(app.say_btn.cget("text") == "Пауза", f"во время чтения кнопка: {app.say_btn.cget('text')}")
        must(not hasattr(app, "say_pause_btn"), "осталась отдельная кнопка паузы")
        app._say_toggle()                              # ❚❚ - та же кнопка
        must(live.paused and app.say_btn.cget("text") == "Продолжить", "❚❚")
        import types
        app.focus_force()
        app.update()
        app._on_space(types.SimpleNamespace(widget=app))   # пробел вне текста - ▶
        must(not live.paused, "пробел не продолжил")
        must(app._on_space(types.SimpleNamespace(widget=inner)) is None, "пробел в тексте перехвачен")
        app._say_toggle()
        app._say_toggle()                              # ▶ на паузе - продолжить
        must(not live.paused, "▶ после паузы")
        app._say_stop()                                # ■
        must(live.stopped and app._live is None, "■ не сработал")
        must(pump(lambda: str(app.say_btn.cget("state")) == "normal"), "кнопка не вернулась")
        app.do_say()                                   # до конца
        must(pump(lambda: app._live is None, 15), "чтение не закончилось")
        must(str(app.say_save_btn.cget("state")) == "normal", "«Сохранить» не включилась")
        must("первый звук" in app.say_note.cget("text"), app.say_note.cget("text"))
    check("вкладка «Прочитать вслух»: одна ▶/❚❚, пробел, ■, подсветка, «Сохранить»", say_flow, limit=60)

    def say_keys():
        """Cmd/Ctrl+A C V X Z - и при русской раскладке (кириллическое имя клавиши)."""
        inner = app.say_box._textbox
        mod = "Command" if sys.platform == "darwin" else "Control"
        app.say_box.delete("1.0", "end")
        app.say_box.insert("1.0", "Проверка буфера")
        inner.focus_force()
        app.update()
        def press(ks):
            # Tk отдаёт клавишу только виджету с фокусом; если человек в это
            # время работает в другом окне, фокус надо вернуть (иначе тест плавает)
            for _ in range(5):
                inner.focus_force()
                app.update()
                if app.focus_get() is inner:
                    break
                time.sleep(0.1)
            inner.event_generate(f"<{mod}-KeyPress>", keysym=ks)
            app.update()
        for ks in ("a", "Cyrillic_ef"):              # выделить всё
            inner.tag_remove("sel", "1.0", "end")
            press(ks)
            must(inner.tag_ranges("sel"), f"{mod}+{ks}: не выделилось")
        app.clipboard_clear()
        press("Cyrillic_es")      # Cmd+С (рус.)
        app.update()
        must(app.clipboard_get() == "Проверка буфера", "Cmd+С при русской раскладке")
        press("Cyrillic_che")     # Cmd+Ч - вырезать
        app.update()
        must(app._say_text() == "", "Cmd+Ч - вырезать")
        press("Cyrillic_em")      # Cmd+М - вставить
        app.update()
        must(app._say_text() == "Проверка буфера", "Cmd+М - вставить")
        press("z")                # отменить вставку
        app.update()
        must(app._say_text() == "", f"Cmd+Z: «{app._say_text()}»")
        must(app._say_key_action(type("E", (), {"keysym": "??", "keycode": 8 << 24 if
                                               sys.platform == "darwin" else 67})()) == "copy",
             "по коду клавиши")
    check("Cmd/Ctrl+A C V X Z при русской и английской раскладке", say_keys)

    def say_voices():
        # 3.6: по умолчанию клонов во вкладке нет, «Быстрый / Клон» и «Начало» скрыты
        app.say_clones_var.set(False)
        app._on_say_clones()
        app.update()
        fast = list(app.say_voice_menu.cget("values"))
        must(any("Айдар" in x for x in fast) and len(fast) >= 5, fast)
        must(all(("муж." in x or "жен." in x) and ("русский" in x or "английский" in x)
                 for x in fast), f"подписи имя · пол · язык: {fast}")
        must(not any("Бая" in x for x in fast), "Бая должна быть скрыта (выбор Андрея 3.7)")
        must(not app.say_mode_box.winfo_ismapped() and not app.say_start_box.winfo_ismapped(),
             "без клонов переключатели видны")
        app._say_on_kind("Клон")
        must(not any("клон" in x for x in app.say_voice_menu.cget("values")), "клоны видны")
        # подсветка: светлая заливка, тёмный текст - контраст по WCAG
        def lum(h):
            c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
            return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
        L1, L2 = sorted((lum(G.SAY_HL_BG), lum(G.SAY_HL_FG)), reverse=True)
        must((L1 + 0.05) / (L2 + 0.05) >= 7, "контраст подсветки ниже 7:1")
        sl0 = app._say_fx_vars["speed"][0]
        must(abs(sl0.cget("from_") - 0.5) < 1e-6 and abs(sl0.cget("to") - 1.5) < 1e-6,
             "темп 0,5…1,5")
        app.say_clones_var.set(True)
        app._on_say_clones()
        app.update()
        must(app.say_mode_box.winfo_ismapped(), "с клонами «Быстрый / Клон» не виден")
        app._say_on_kind("Быстрый")
        app._say_on_kind("Клон")
        clones = list(app.say_voice_menu.cget("values"))
        must(clones and not any("Айдар" in x for x in clones), clones)
        app.update()
        must(app.say_start_box.winfo_ismapped(), "для клона «Начало» не видно")
        want = clones[-1]
        app.say_voice_var.set(want)
        app._say_on_voice()
        app._say_on_kind("Быстрый")
        app._say_on_kind("Клон")
        must(app.say_voice_var.get() == want, "выбор клона не запомнился")
        # ползунок темпа - прогноз пересчитывается сразу
        app.say_box.delete("1.0", "end")
        app.say_box.insert("1.0", "Море к вечеру стихло. " * 30)
        app._say_text_changed()
        sl = app._say_fx_vars["speed"][0]
        sl.set(0.8)
        app._say_fx_changed("speed")
        slow = app.say_forecast.cget("text")
        sl.set(1.2)
        app._say_fx_changed("speed")
        must(app.say_forecast.cget("text") != slow, "прогноз не пересчитан по темпу")
        app._say_fx_reset()
        must(abs(sl.get() - 1.0) < 1e-6, "«Сбросить»")
        app._say_on_kind("Быстрый")
        app.say_clones_var.set(False)
        app._on_say_clones()
    check("голоса: клоны скрыты, подписи, темп 0,5-1,5, светлая подсветка, прогноз", say_voices)

    def stream_locked():
        app._set_busy(True, "тест")
        off = [b for b in [app.say_btn, app.fx_play_btn] + list(app._prof_btns.values())
               if str(b.cget("state")) != "disabled"]
        app._set_busy(False)
        must(not off, f"во время счёта не заблокировано: {len(off)}")
        must(str(app.say_btn.cget("state")) == "normal", "после счёта не вернулось")
    check("во время озвучки книги прослушивание заперто", stream_locked)

    def preview_stream():
        app._set_book(book)
        pump(lambda: app.est is not None)
        app.do_preview()
        live = app._live
        must(app._live_kind == "preview" and live is not None, "пробник не потоком")
        must(str(app.pause_btn.cget("state")) == "normal", "пауза недоступна")
        must(pump(lambda: live.first_sound is not None), "звук пробника не пошёл")
        must(pump(lambda: "Пробник · " in app.status.cget("text")), app.status.cget("text"))
        app.do_pause()
        must(live.paused and app.pause_btn.cget("text") == "Продолжить", "пауза звука")
        app.do_pause()
        must(not live.paused, "продолжить")
        app.do_stop()
        must(live.stopped, "«Остановить» не оборвал звук")
        must(pump(lambda: not app.running, 15), "пробник не закончился")
    check("пробник потоком: пауза, «Остановить»", preview_stream, limit=60)

    def profile_listen_needs_clone():
        silero = next(v for v in app._voices if v.ready)
        keep = app.voice_var.get()
        app.voice_var.set(silero.label())
        app._profile_listen("main")
        app.voice_var.set(keep)
        must("клон" in app.prof_note.cget("text"), app.prof_note.cget("text"))
    check("«Послушать» движок - на клоне", profile_listen_needs_clone)

    def server():
        import json as _json
        import socket
        import urllib.request
        from core import speak_server
        speak_server.SpeakServer._warm = lambda self: None   # без загрузки моделей
        sk = socket.socket()
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
        sk.close()
        app.srv_port.delete(0, "end")
        app.srv_port.insert(0, str(port))
        app.srv_var.set(True)
        app._srv_toggle()
        must(app._srv is not None and app._srv.running, app.srv_note.cget("text"))
        r = _json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5))
        must(r.get("ok"), r)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health",
                                     headers={"Host": "evil.example"})
        try:
            urllib.request.urlopen(req, timeout=5)
            must(False, "чужой Host пропущен")
        except urllib.error.HTTPError as e:
            must(e.code == 403, e.code)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health",
                                     headers={"Origin": "https://evil.example"})
        try:
            urllib.request.urlopen(req, timeout=5)
            must(False, "чужой сайт пропущен")
        except urllib.error.HTTPError as e:
            must(e.code == 403, e.code)
        app.srv_var.set(False)
        app._srv_toggle()
        must(app._srv is None, "сервис не выключился")
    check("сервис /speak: только этот компьютер, включается и выключается", server)

    def read_text():
        """«Как будет прочитано»: ударение знаком, только просмотр (3.8)."""
        keep_prep = P.prepare_read_text
        P.prepare_read_text = lambda path, opts: "=== Часть 1 ===\n\nОни вз+ялись за дело.\n"
        real_path, real_disp = P.read_text_path, P.read_disputed
        P.read_text_path = lambda path, out_dir=None: "/нет/такого/файла"
        P.read_disputed = lambda path, out_dir=None: ["взялись"]
        try:
            app._set_book(book)
            app.show_read_text()
            must(pump(lambda: getattr(app, "_read_win", None) is not None
                      and app._read_win.winfo_exists(), 10), "окно не открылось")
            box = app._read_win._box
            body = box.get("1.0", "end")
            must("вз\u044f\u0301лись" in body, f"ударение знаком: {body!r}")
            must(str(box.cget("state")) == "disabled", "текст можно править")
            must(not hasattr(app._read_win, "_fix"), "осталось «Исправить ударение»")
            must(app._read_win._disputed == 1, "спорное слово (RUAccent против silero) не подсвечено")
        finally:
            P.prepare_read_text, P.read_text_path, P.read_disputed = keep_prep, real_path, real_disp
            if getattr(app, "_read_win", None) is not None:
                app._read_win.destroy()
    check("«Как будет прочитано»: ударения знаком, спорные подсвечены, только просмотр", read_text)

    def busy_gpu():
        """Честный прогноз во время работы и «видеокарта занята» (3.7)."""
        from core import busy
        real_owner, real_free = busy.lock_owner, busy.lock_free
        app.show("book")
        app._set_busy(True, "тест")
        try:
            app._step = 2
            t = time.time()
            # быстро в начале, потом в 3 раза медленнее - как при чужой нагрузке
            app._prog_hist = [(t - 400, 0.0), (t - 300, 0.2), (t - 170, 0.25), (t - 10, 0.3)]
            app._rate_base = None
            rest = app._eta()
            must(rest is not None and rest > 1500, f"прогноз по недавней скорости: {rest}")
            app.status.configure(text="Шаг 2 из 3 · Озвучка - часть 1, 30 % · осталось ≈ 9 мин")
            busy.lock_owner = lambda: "Понятный документ"
            busy.lock_free = lambda: False
            app._eta_refresh()
            app.update()
            must(app.busy_line.winfo_ismapped(), "строка «видеокарта занята» не появилась")
            must("«Понятный документ»" in app.busy_note.cget("text"), app.busy_note.cget("text"))
            must("осталось ≈" in app.status.cget("text") and "9 мин" not in app.status.cget("text"),
                 f"прогноз не обновился: {app.status.cget('text')}")
            app._wait_free()
            must(app._pause_flag.is_set() and app._wait_lock, "«Подождать» не поставил паузу")
            busy.lock_free = lambda: True
            busy.lock_owner = lambda: None
            must(pump(lambda: not app._pause_flag.is_set(), 15), "после снятия замка не продолжил")
            app.update()
            must(not app.busy_line.winfo_ismapped(), "строка занятости осталась")
        finally:
            busy.lock_owner, busy.lock_free = real_owner, real_free
            app._set_busy(False)
    check("прогноз по недавней скорости, «видеокарта занята», «Подождать»", busy_gpu, limit=40)

    def forecast_cache():
        """3.8: куски книги уже в кэше - прогноз меньше и говорит об этом."""
        app._set_book(book)
        pump(lambda: app.est is not None)
        app._cache_key = None
        real = G.cached_share
        G.cached_share = lambda *a, **k: None
        app.show_estimate()
        pump(lambda: getattr(app, "_cache_key", None) is not None, 2)
        app.update()
        full = app.est_label.cget("text")
        G.cached_share = lambda *a, **k: {"n": 15, "cached": 12, "chars": 1500, "cached_chars": 1200}
        app._cache_key = None
        try:
            app.show_estimate()
            must(pump(lambda: "12 из 15" in app.est_label.cget("text"), 5), app.est_label.cget("text"))
            import re as _r
            def wait_sec(t):
                m = _r.search(r"Ждать около ([^·]+)", t)
                return m.group(1).strip() if m else ""
            must(wait_sec(app.est_label.cget("text")) != wait_sec(full),
                 f"ожидание не уменьшилось: {wait_sec(full)} -> {wait_sec(app.est_label.cget('text'))}")
        finally:
            G.cached_share = real
            app._cache_key = None
    check("прогноз: куски из кэша - «12 из 15 кусков уже посчитаны», ждать меньше", forecast_cache)

    def voice_manager():
        """«Голоса» (3.8): каждый голос - строкой; послушать, скачать (с ходом),
        нет интернета - понятная ошибка, удалить - через «⋯» с подтверждением,
        Vosk - скрыть один, все три скрыты - удалить пакет; своё имя (✎). В
        «Настройках» голосов больше нет. Без сети."""
        from core import voice_store as VS
        from tkinter import messagebox as mb
        app.show("settings")
        app.update()
        must(not hasattr(app, "vm_rows"), "в Настройках осталась карточка голосов")
        app.show("voices")
        app.update()
        rows = app._voice_row_widgets
        for vid in [c.id for c in VS.CATALOG]:
            must(vid in rows, f"нет строки голоса {vid}")
        def labels(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkLabel):
                    yield c.cget("text")
                yield from labels(c)
        texts = " | ".join(t for r in rows.values() for t in labels(r))
        must("входит в пакет Vosk" in texts and "Joe" in texts, texts[-300:])
        keep_cfg = {k: app.cfg.get(k) for k in ("hidden_voices", "voice_names")}
        app._voice_action("eugene", "Евгений (муж.)", "hide")          # «⋯» → «Скрыть»
        must(not any(v.id == "eugene" for v in app._voices), "скрытый голос остался в списке")
        must(("show", "Показать в списках") in rows["eugene"]._herald_menu
             or any(a == "show" for a, _t in app._voice_row_widgets["eugene"]._herald_menu),
             "у скрытого нет «Показать»")
        app._voice_action("eugene", "Евгений (муж.)", "show")
        must(any(v.id == "eugene" for v in app._voices), "голос не вернулся")
        # своё имя: во всех списках, исходное - рядом
        app._rename_voice("aidar", "Айдар (муж.)")
        app.update()
        app._rename_dlg._save("Диктор А")
        must(any(v.id == "aidar" and v.title == "Диктор А" for v in app._voices),
             "своё имя не дошло до списков")
        must(any("Диктор А" in x for x in app.voice_menu.cget("values")), "в меню голоса старое имя")
        app._rename_voice("aidar", "Айдар (муж.)")
        app.update()
        app._rename_dlg._save("")
        must(any(v.id == "aidar" and v.title.startswith("Айдар") for v in app._voices),
             "исходное имя не вернулось")
        joe = VS.by_id("piper:en_US-joe-medium")
        real_dl, real_ok, real_yes, real_del = VS.download, mb.askokcancel, mb.askyesno, VS.delete
        real_inst = VS.installed
        mb.askokcancel = mb.askyesno = lambda *a, **k: True
        try:
            def fake(v, progress=None, cancel=None):
                for i in range(5):
                    progress(i / 4, f"Скачиваю {v.title}: {i} из 4 МБ")
                    time.sleep(0.05)
            VS.download = fake
            app._vm_download_id(joe.id)
            must(pump(lambda: app._vm_busy is None, 10), "загрузка не закончилась")
            must("Готово" in app.vm_note.cget("text"), app.vm_note.cget("text"))

            def offline(v, progress=None, cancel=None):
                raise VS.NoInternet("Нет связи с интернетом - голос не скачался. Проверьте "
                                    "подключение и нажмите «Скачать» ещё раз.")
            VS.download = offline
            app._vm_download_id(joe.id)
            must(pump(lambda: app._vm_busy is None, 10), "ошибка не пришла")
            must("Нет связи с интернетом" in app.vm_note.cget("text"), app.vm_note.cget("text"))
            deleted = []
            VS.delete = lambda v: deleted.append(v.id)
            VS.installed = lambda v: True
            app._voice_action(joe.id, joe.title, "delete")
            must(deleted == [joe.id], "удаление Piper не спросило или не удалило")
            deleted.clear()
            for vid in ("vosk:4", "vosk:3"):
                app._voice_action(vid, vid, "delete")
            must(not deleted and {"vosk:4", "vosk:3"} <= app._hidden_voices(),
                 "голос Vosk не скрылся или пакет удалён раньше времени")
            app._voice_action("vosk:2", "vosk:2", "delete")
            must(deleted == ["vosk:2"], f"все три Vosk скрыты - пакет не удалён: {deleted}")
        finally:
            VS.download, mb.askokcancel, mb.askyesno, VS.delete = real_dl, real_ok, real_yes, real_del
            VS.installed = real_inst
            for k, v in keep_cfg.items():
                if v is None:
                    app.cfg.pop(k, None)
                else:
                    app.cfg[k] = v
            app.refresh_voices()
            app._refresh_voice_rows()
    check("«Голоса»: строка на голос, послушать, скачать, нет сети, «⋯» удалить, Vosk, своё имя", voice_manager, limit=60)

    G.S.speak, G.S.default_sink, G.preview_sample = real_speak, real_sink, real_preview

    G.save_config(cfg_backup)
    app.destroy()

    print("язык интерфейса EN и стандарт (3.7):")
    from core import i18n
    import re as _re

    def walk(w, out):
        for c in w.winfo_children():
            if getattr(c, "_herald_data", False):
                continue
            for opt in ("text",):
                try:
                    v = c.cget(opt)
                    if isinstance(v, str):
                        out.append((type(c).__name__, v))
                except Exception:
                    pass
            if isinstance(c, (ctk.CTkSegmentedButton, ctk.CTkOptionMenu)):
                try:
                    out += [(type(c).__name__, str(x)) for x in c.cget("values")]
                except Exception:
                    pass
            walk(c, out)
        return out

    def english():
        i18n.set_lang("en")
        try:
            e = G.App()
            e.update()
            clones = sorted({x for v in e._voices if not v.ready
                             for x in (v.title, G._short(v.title, G.VOICE_NAME_LIMIT), v.id)},
                            key=len, reverse=True)
            found = []
            for key in ("book", "say", "voices", "help", "settings"):
                e.show(key)
                e.update()
            texts = walk(e, [])
            # тексты справки и «О программе» - тоже интерфейс
            for page in e.pages.values():
                for c in page.winfo_children() + [x for p_ in page.winfo_children()
                                                   for x in p_.winfo_children()]:
                    if isinstance(c, ctk.CTkTextbox) and c not in (e.say_box, e.log):
                        texts.append(("CTkTextbox", c.get("1.0", "end")))
            menu = e.nametowidget(e.cget("menu"))
            for i in range(menu.index("end") + 1):
                texts.append(("Menu", menu.entrycget(i, "label")))
                sub = e.nametowidget(menu.entrycget(i, "menu"))
                for j in range((sub.index("end") or 0) + 1):
                    if sub.type(j) == "command":
                        texts.append(("Menu", sub.entrycget(j, "label")))
            for kind, t in texts:
                for cid in clones:                         # имена клонов - данные пользователя
                    t = t.replace(cid, "")
                if _re.search(r"[А-Яа-яЁё]", t):
                    found.append(f"{kind}: {t.strip()[:80]!r}")
            must(not found, f"кириллица при EN ({len(found)}): " + " | ".join(found[:40]))
            must(e.lang_seg.get() == "EN", "переключатель языка не EN")
            must(abs(e._scale - (G.load_config().get("scale") or G.DEFAULT_SCALE)) < 1e-6
                 and G.DEFAULT_SCALE == 1.2, "масштаб по умолчанию не 120 %")
            must(G.VERSION in e.version_lbl.cget("text"), "версия внизу не совпадает с пакетом")
            th = e.cfg.get("theme", "system")
            e._cycle_theme()
            must(e.cfg.get("theme") != th, "тема не переключается")
            e.set_theme(th)
            e.destroy()
        finally:
            i18n.set_lang("ru")
    check("EN: в окне нет кириллицы, масштаб 120 %, версия, тема", english, limit=120)

    def no_missing():
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        import i18n_keys
        miss = i18n.missing(i18n_keys.all_keys())
        must(not miss, f"ключи без перевода ({len(miss)}): {miss[:5]}")
    check("в словаре EN нет пропущенных ключей", no_missing)
    print("всё в порядке" if not FAILS else f"\nПРОВАЛЕНО: {len(FAILS)}")
    for f in FAILS:
        print("   ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
