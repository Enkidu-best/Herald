#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон окна без человека: открыть, потыкать, поймать ошибки (окно 3.1).

Родился из истории, где одну и ту же кнопку ломали трижды подряд, а замечал
это только пользователь. Тест поднимает настоящее окно, дёргает обработчики и
ловит всё, что упало, - включая исключения из колбэков tkinter, которые иначе
просто печатаются в консоль и никого не останавливают.

    python "Инструменты разработки/test_gui.py"              # без моделей
    python "Инструменты разработки/test_gui.py" --shuffle [N]  # проверки 12-14 в случайном порядке
    python "Инструменты разработки/test_gui.py" --warm         # с прогревом Silero (под замком)

Синтез здесь НЕ запускается: проверяется интерфейс, а не звук.

Данные - свои (стандарт 6.10, задание 13 Б): временная папка HERALD_DATA_DIR с двумя
клонами, примером книги и пустой корзиной (`herald_testdata.py`); голоса, настройки и
корзина Андрея не читаются и не меняются - в конце сверяются хэши. Проверки, которым
нужен Mac (звук, раскладка), на других машинах пропускаются с причиной.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(1, _os.path.dirname(_os.path.abspath(__file__)))
_os.chdir(_sys.path[0])
import herald_testdata as TD          # до core и tts_gui: HERALD_DATA_DIR
TD.setup("herald_gui_")

import os
import sys
import time
import traceback

os.environ.setdefault("PYTHONHASHSEED", "0")

FAILS: list[str] = []
SKIPS: list[str] = []
# --shuffle [зерно]: проверки заданий 12-14 - в случайном порядке (задание 14 А: проверка не
# должна зависеть от того, что шло перед ней). Зерно печатается - порядок можно повторить.
SHUFFLE = "--shuffle" in sys.argv
DEFER: list | None = None


def check(name: str, fn, limit: int = 30, skip: str | None = None):
    import signal
    if DEFER is not None:
        DEFER.append((name, fn, limit, skip))
        return
    if skip:
        SKIPS.append(f"{name}: {skip}")
        print(f"  ПРОПУСК {name}: {skip}")
        return
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

    # Модели окну не нужны (синтез не запускается). Фоновый прогрев «Прочитать вслух» и
    # сервиса качал и грузил Silero и расстановку ударений: на медленной машине без
    # моделей он отнимал процессор у других шагов, и проверка зависела от порядка
    # (задание 14 А). По умолчанию - выключен; любая загрузка модели - ошибка проверки.
    # --warm - как раньше, с прогревом (под замком видеокарты).
    if "--warm" not in sys.argv:
        from core import speak_server as _ss

        def _no_model(*_a, **_k):
            raise RuntimeError("проверка окна попробовала загрузить модель")
        _ss.SpeakServer._warm = lambda self: None
        G.App._say_warm = lambda self: None
        P._loaded_engine = _no_model
        from core import asr as _asr
        _asr.transcribe = _no_model               # Whisper - тоже модель (видеокарта на Mac)
        print("модели и прогрев выключены (--warm - включить)")

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
    book = TD.BOOK
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
            # движок клона - по компьютеру: Mac - f5mlx, Linux / Windows - f5torch
            must(engine == G.clone_engine() and kw == profiles.engine_kwargs(profiles.get(key), app._quality()),
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
            must(q in app._rtf_key() and G.SPEED_SET in app._rtf_key(), app._rtf_key())
        app.quality_var.set(cfg_backup.get("quality", "balanced"))
        app._on_quality()
        ready = next(v for v in app._voices if v.ready and v.engine == "silero")   # задание 12: по алфавиту первым может быть Vosk
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
        old = app.cfg.get("rtf", {}).get(app._rtf_key())
        app._set_busy(True, "тест")
        app._got_rtf(2.05)
        app._show_progress(0.3, "Озвучиваю: часть 1 из 2")
        t = app.status.cget("text")
        app._set_busy(False)
        must("RTF 2,05" in t, t)
        # замер сглаживается с прежним (0,7 × старое + 0,3 × новое): при настоящем прежнем
        # замере на этом Mac ждём именно сглаженное значение, а не 2,05
        want = 2.05 if not old else 0.7 * old + 0.3 * 2.05
        must(abs(app.cfg["rtf"][app._rtf_key()] - want) < 0.01, (old, app.cfg.get("rtf")))
        if old:
            app.cfg["rtf"][app._rtf_key()] = old          # настоящий замер Андрея - как был
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
    check("мини-плеер: играть/пауза, стоп, время, перемотка, одна часть за раз", mini_player,
          skip=TD.need_mac("плеер частей - системный AVAudioPlayer и звуковое устройство"))

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
    check("Cmd/Ctrl+A C V X Z при русской и английской раскладке", say_keys,
          skip=TD.need_mac("коды клавиш и русская раскладка macOS"))

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
        # задание 12: ползунков звучания и A− / A+ здесь нет - звучание одно, на «Голосах»
        must(not hasattr(app, "_say_fx_vars") and not hasattr(app, "say_zoom_lbl"),
             "на «Прочитать вслух» остались ползунки звучания или A− / A+")
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
        # темп сохранён для голоса на «Голосах» - строка звучания и прогноз его видят
        app.say_box.delete("1.0", "end")
        app.say_box.insert("1.0", "Море к вечеру стихло. " * 30)
        app._say_text_changed()
        v = app._say_voice()
        keep = load_fx(default_voices_dir(), v.id)
        slow = app.say_forecast.cget("text")
        try:
            fx = load_fx(default_voices_dir(), v.id)
            fx.speed = 1.2
            save_fx(default_voices_dir(), v.id, fx)
            app._say_on_voice(remember=False)
            must("1,20" in app.say_fx_line.cget("text"), app.say_fx_line.cget("text"))
            must(app.say_forecast.cget("text") != slow, "прогноз не пересчитан по темпу")
        finally:
            save_fx(default_voices_dir(), v.id, keep)
        app._say_on_voice(remember=False)
        app._open_voice_fx(app._say_voice())
        must(app.page == "voices" and app._fx_voice == v.id, "«Звучание голоса ›» не открыла голос")
        app.show("say")
        app._say_on_kind("Быстрый")
        app.say_clones_var.set(False)
        app._on_say_clones()
    check("голоса: клоны скрыты, подписи, звучание - ссылкой на «Голоса», светлая подсветка, прогноз", say_voices)

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
        # задание 13 Г5: голос не Silero - ошибка с причиной, а не молча «Айдар»
        req = urllib.request.Request(f"http://127.0.0.1:{port}/speak", method="POST",
                                     data=_json.dumps({"text": "Проверка.", "fast": True,
                                                       "voice": "vosk:4"}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
            must(False, "голос Vosk молча подменён")
        except urllib.error.HTTPError as e:
            body = _json.loads(e.read().decode("utf-8"))
            must(e.code == 400 and "не озвучивает" in body.get("error", ""), (e.code, body))
        # при «Qwen ru-stress» в окне клон через сервис - параметрами «Основного» (F5)
        from core import profiles as _pr
        keep_p = app.profile_var.get()
        if _pr.available(_pr.PROFILES["qwen"]):
            app.profile_var.set("qwen")
            app._on_profile()
            must(app._srv_kwargs == _pr.engine_kwargs(_pr.PROFILES["main"], app._quality()),
                 f"сервис при Qwen: {app._srv_kwargs}")
            app.profile_var.set(keep_p)
            app._on_profile()
        app.srv_var.set(False)
        app._srv_toggle()
        must(app._srv is None, "сервис не выключился")
    check("сервис /speak: только этот компьютер, включается и выключается; не Silero - 400 с причиной", server)

    def not_applicable():
        """Г4: «Качество» и «Проверка» говорят, где не действуют; strict_words не пишется."""
        from core import profiles as _pr
        keep_v, keep_p = app.voice_var.get(), app.profile_var.get()
        try:
            silero = next(v for v in app._voices if v.ready and v.engine == "silero")
            app.voice_var.set(silero.label())
            app._on_voice(remember=False)
            must("Не действует для голоса" in app.quality_note.cget("text"), app.quality_note.cget("text"))
            must("Не действует для голоса" in app.sound_note.cget("text"), app.sound_note.cget("text"))
            must(all(str(r.cget("state")) == "disabled" for r in app._quality_radios), "качество не серое")
            clone = next(v for v in app._voices if v.id == TD.CLONE)
            app.voice_var.set(clone.label())
            app._on_voice(remember=False)
            must(app.quality_note.cget("text") == "" and app.sound_note.cget("text") == "", "у клона - «не действует»")
            must(all(str(r.cget("state")) == "normal" for r in app._quality_radios), "у клона качество серое")
            app.profile_var.set("draft")
            app._on_profile()
            must("Черновик" in app.quality_note.cget("text"), app.quality_note.cget("text"))
            app._on_strict()
            must("strict_words" not in G.load_config(), "strict_words снова пишется в настройки")
        finally:
            app.profile_var.set(keep_p)
            app._on_profile()
            app.voice_var.set(keep_v)
            app._on_voice(remember=False)
    check("«Качество» и «Проверка» - «не действует» для готового голоса и «Черновика»", not_applicable)

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
        app.show_hidden_var.set(True)              # задание 12: скрытые - по флажку
        app._refresh_voice_rows()
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
            app.show_hidden_var.set(False)
            app.refresh_voices()
            app._refresh_voice_rows()
    check("«Голоса»: строка на голос, послушать, скачать, нет сети, «⋯» удалить, Vosk, своё имя", voice_manager, limit=60)

    global DEFER
    if SHUFFLE:
        DEFER = []
    print("задание 12:")

    def voice_order():
        """Порядок: «Мои голоса», «Русские», «Английские», по алфавиту; Qwen - в русских."""
        app.show("voices")
        app._refresh_voice_rows()
        app.update()
        heads = [c.cget("text") for c in app.voice_rows.winfo_children()
                 if isinstance(c, ctk.CTkLabel) and " · " in str(c.cget("text"))]
        must(heads and heads[0].startswith("Мои голоса · "), f"группы: {heads}")
        must([h.split(" · ")[0] for h in heads][:3] == ["Мои голоса", "Русские", "Английские"], heads)
        vs = app._voices
        mine = [v for v in vs if v.mine]
        must(vs[:len(mine)] == mine, "«Мои голоса» не первые в списке выбора")
        ru = [G._(v.title).lower() for v in vs if v.group() == "ru"]
        must(ru == sorted(ru), "русские не по алфавиту")
        if app._qwen_ok():
            q = [v for v in vs if v.qwen]
            must(q and all(v.group() == "ru" for v in q), "голоса Qwen не в русских")
            must(all(" · Qwen" in v.label() for v in q), "у голоса Qwen нет пометки")
            eng, kw = None, None
            keep = app.voice_var.get()
            app.voice_var.set(q[0].label())
            eng, kw = app._eng_choice()
            app.voice_var.set(keep)
            must(eng == "mlxaudio", f"голос Qwen читает не Qwen: {eng}")
        must(not any(v.hidden for v in vs), "скрытый голос в списке выбора")
    check("порядок голосов: «Мои голоса», русские, английские; Qwen - в русских", voice_order)

    def voice_trash():
        """«Удалить…» клона - в корзину, «Вернуть» - назад; «Удалить навсегда» - из корзины."""
        import shutil as _sh
        from core import trash
        from tkinter import messagebox as mb
        vd = default_voices_dir()
        src = next(v.id for v in app._all_voices() if not v.ready and not v.qwen)
        name = "Тест корзины 12"
        for e in (".wav", ".txt"):
            _sh.copy(os.path.join(vd, src + e), os.path.join(vd, name + e))
        real_yes = mb.askyesno
        mb.askyesno = lambda *a, **k: True
        try:
            app.refresh_voices()
            must(any(v.id == name for v in app._voices), "тестовый голос не появился")
            app._voice_trash(name)
            must(not any(v.id == name for v in app._all_voices()), "голос не удалился")
            it = next((x for x in trash.items() if x.title == name), None)
            must(it is not None, "в корзине нет голоса")
            must("Вернуть" == app.vm_undo.cget("text"), "нет «Вернуть» после удаления")
            app.vm_undo.invoke()
            must(any(v.id == name for v in app._voices), "голос не вернулся из корзины")
            app._voice_trash(name)
            it = next(x for x in trash.items() if x.title == name)
            app.show("settings")
            app._refresh_trash()
            app._trash_purge(it)
            must(not any(x.id == it.id for x in trash.items()), "«Удалить навсегда» не удалил")
        finally:
            mb.askyesno = real_yes
            for e in (".wav", ".txt", ".src.json", ".fx.json"):
                try:
                    os.remove(os.path.join(vd, name + e))
                except OSError:
                    pass
            for x in trash.items():
                if x.title == name:
                    trash.purge(x)
            app.refresh_voices()
            app._refresh_voice_rows()
    check("«Удалить…» клона - в корзину, «Вернуть», «Удалить навсегда»", voice_trash)

    def aa_popover():
        """«Aa» - одна панель в окне (не окно), повторное нажатие закрывает; 80-200 %."""
        import tkinter as _tk
        keep = app._scale
        tops = lambda: [w for w in app.winfo_children() if isinstance(w, _tk.Toplevel)]
        n0 = len(tops())
        try:
            for _i in range(5):
                app._aa_popup()
                app.update()
            must(len(tops()) == n0, "«Aa» открывает отдельное окно")
            must(app._aa_pop.winfo_ismapped(), "после 5 нажатий панель должна быть открыта")
            app._aa_popup()
            app.update()
            must(not app._aa_pop.winfo_ismapped(), "повторное «Aa» не закрыло панель")
            for _i in range(20):
                app._scale_by(0.1)
            must(abs(app._scale - 2.0) < 1e-6, f"предел сверху: {app._scale}")
            for _i in range(20):
                app._scale_by(-0.1)
            must(abs(app._scale - 0.8) < 1e-6, f"предел снизу: {app._scale}")
            app._scale_by(None)
            must(abs(app._scale - G.DEFAULT_SCALE) < 1e-6, "«Сбросить» - не 120 %")
            must(app.bind_all("<Command-equal>") or app.bind_all("<Control-equal>"), "нет Cmd+=")
        finally:
            app._set_scale(keep)
            app._aa_close()
            app.update()
    check("«Aa»: одна панель в окне, повторное нажатие закрывает, 80-200 %, Cmd+= / − / 0", aa_popover)

    def book_no_scroll():
        """«Озвучка»: правая панель целиком видна при 1440×900, масштаб 100 и 120 %."""
        keep, geo = app._scale, app.geometry()
        try:
            app.show("book")
            for k in (1.0, 1.2):
                app._set_scale(k)
                app.geometry("1440x810")              # 1440×900 без строки меню и Dock
                app.update()
                time.sleep(0.2)
                app.update()
                o = app.opts_box
                must(o.winfo_reqheight() <= o.winfo_height() + 2,
                     f"масштаб {k:.0%}: панель {o.winfo_reqheight()} px, видно {o.winfo_height()} px")
                must(not isinstance(o, ctk.CTkScrollableFrame), "панель с прокруткой")
        finally:
            app._set_scale(keep)
            app.geometry(geo)
            app.update()
    check("«Озвучка» без прокрутки при 1440×900 и 100 / 120 %", book_no_scroll,
          skip=TD.need_mac("требование - для экрана Mac 1440×900 и его шрифтов (на Linux шрифты крупнее)"))

    def formats():
        """mp3 64-320 и FLAC в списке, с размером часа; оценка места по битрейту."""
        vals = app.fmt_menu.cget("values")
        must(len(vals) == 6 and all("МБ/ч" in v for v in vals), vals)
        must(app._fmt() in G.FORMATS, app._fmt())
        app._set_book(book)
        wait(lambda: app.est is not None)
        keep = app.fmt_var.get()
        sizes = {}
        for key in ("mp3 192 кбит/с (обычный)", "mp3 64 кбит/с (речь, моно, самый маленький файл)"):
            app.fmt_var.set(app._fmt_label(key))
            app.show_estimate()
            sizes[key] = app.est_label.cget("text")
        app.fmt_var.set(keep)
        app._remember()
        must(len(set(sizes.values())) == 2, f"оценка места не зависит от формата: {sizes}")
        must("." not in app.est_label.cget("text").split("на диске")[-1].split()[0].strip("~"),
             "размер с точкой")
    check("формат: mp3 320 / 192 / 128 / 96 / 64 и FLAC, размер часа, оценка места", formats)

    def service_files():
        """Строка «Служебные файлы» в карточке книги, «Очистить» - в корзину."""
        from core import trash
        from tkinter import messagebox as mb
        d = os.path.join(P.output_dir_for(book), P.SERVICE_DIR)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "проба.txt"), "w").write("x" * 4096)
        real_yes = mb.askyesno
        mb.askyesno = lambda *a, **k: True
        try:
            app._set_book(book)
            must(pump(lambda: app.svc_line.winfo_manager() != "", 10), "строки «Служебные файлы» нет")
            must("Служебные файлы" in app.svc_lbl.cget("text"), app.svc_lbl.cget("text"))
            app._service_clear()
            must(not os.path.exists(d), "служебные файлы не убрались")
            must(any(x.kind == "service" for x in trash.items()), "в корзине их нет")
        finally:
            mb.askyesno = real_yes
            for x in trash.items():
                if x.kind == "service" and x.title == os.path.basename(P.output_dir_for(book)):
                    trash.purge(x)
            import shutil as _sh
            _sh.rmtree(P.output_dir_for(book), ignore_errors=True)
    check("служебные файлы: строка, «Очистить» - в корзину", service_files)

    def recent_and_restart():
        """«Недавние книги» в пустой карточке; «Начать заново» спрашивает."""
        from tkinter import messagebox as mb
        keep = list(app.cfg.get("recent_books") or [])
        try:
            app._set_book(book)
            must(app._recent_list()[0] == os.path.abspath(book), app._recent_list())
            app.clear_book()
            app.update()
            texts = [c.cget("text") for c in app.recent_box.winfo_children() if hasattr(c, "cget")]
            must(any("Недавние книги" in str(t) for t in texts), texts)
            must(str(app.start_btn.cget("state")) == "disabled", "без книги «Озвучить книгу» активна")
            must(not app.pause_btn.winfo_manager(), "«Пауза» видна без озвучки")
            app._set_book(book)
            asked, real_yes = [], mb.askyesno
            mb.askyesno = lambda *a, **k: (asked.append(a), False)[1]
            forgot = []
            real_forget = G.forget_progress
            G.forget_progress = lambda p: forgot.append(p)
            app._resume = {"done": 1, "total": 3}
            app.do_restart()
            mb.askyesno, G.forget_progress = real_yes, real_forget
            must(asked and not forgot, "«Начать заново» не спросило или стёрло без согласия")
        finally:
            app.cfg["recent_books"] = keep
            app._resume = None
    check("«Недавние книги», одна главная кнопка, «Пауза» - только в работе, «Начать заново» спрашивает",
          recent_and_restart)

    def stress_fix():
        """«Как будет прочитано»: щелчок по спорному слову - ударение для этой книги."""
        from core import bookfix
        keep_prep = P.prepare_read_text
        P.prepare_read_text = lambda path, opts: "=== Часть 1 ===\n\nСтарый з+амок и новый з+амок.\n"
        real_path, real_disp = P.read_text_path, P.read_disputed
        P.read_text_path = lambda path, out_dir=None: "/нет/такого/файла"
        P.read_disputed = lambda path, out_dir=None: ["замок"]
        keep_fix = bookfix.load(book)
        try:
            app._set_book(book)
            app.show_read_text()
            must(pump(lambda: getattr(app, "_read_win", None) is not None
                      and app._read_win.winfo_exists()
                      and getattr(app._read_win, "_disputed", 0) == 2, 10), "окно не открылось")
            bookfix.set_fix(book, "замок", 3)           # как выбор «Ударение: замо́к»
            app._read_win._render()
            body = app._read_win._box.get("1.0", "end")
            must(body.count("замо\u0301к") == 2 and app._read_win._fixed == 1, repr(body))
            must(bookfix.apply("в з+амке и з+амок", bookfix.load(book)) == "в з+амке и зам+ок",
                 "поправка задела другое слово или не встала")
            app._read_win.destroy()
        finally:
            bookfix.set_fix(book, "замок", keep_fix.get("замок"))
            P.prepare_read_text, P.read_text_path, P.read_disputed = keep_prep, real_path, real_disp
    check("«Как будет прочитано»: поправка спорного ударения для книги", stress_fix)

    print("задание 13: «Голоса», замок видеокарты, закрытие, «голос изменился»:")
    from tkinter import messagebox as _mb
    from core import heavylock as HL
    import json as _json
    import shutil as _shutil

    def menu_while_busy():
        """Б1: [≡] во время озвучки - сообщение, меню и настройка не меняются."""
        said = []
        real_info = _mb.showinfo
        _mb.showinfo = lambda *a, **k: said.append(a)
        keep = app.cfg.get("menu_collapsed", False)
        try:
            app._set_busy(True, "тест")
            app._toggle_menu()
            must(said and "озвучка закончится" in str(said[0]), f"нет сообщения: {said}")
            must(app.cfg.get("menu_collapsed", False) == keep, "выбор записался во время озвучки")
        finally:
            _mb.showinfo = real_info
            app._set_busy(False)
    check("[≡] во время озвучки - «Свернуть меню можно, когда озвучка закончится»", menu_while_busy)

    def one_hint():
        """Б2: пояснение - одна строка для выбранного движка и качества."""
        from core import profiles as _pr
        keep_p, keep_q = app.profile_var.get(), app.quality_var.get()
        texts = []

        def walk(w):
            for c in w.winfo_children():
                if isinstance(c, ctk.CTkLabel):
                    texts.append(c.cget("text"))
                walk(c)
        walk(app.pages["settings"])
        hints = [_pr.PROFILES[k].hint for k in _pr.PROFILES] + [q.hint for q in _pr.QUALITY.values()]
        shown = [t for t in texts if t in hints]
        must(len(shown) == 2, f"пояснений на странице {len(shown)}, нужно 2 (движок и качество)")
        try:
            for key in ("draft", "main"):
                app.profile_var.set(key)
                app._on_profile()
                must(app.prof_hint.cget("text") == _pr.PROFILES[key].hint, app.prof_hint.cget("text"))
            for q in _pr.QUALITY.values():
                app.quality_var.set(q.key)
                app._on_quality()
                must(app.quality_hint.cget("text") == q.hint, app.quality_hint.cget("text"))
        finally:
            app.profile_var.set(keep_p)
            app._on_profile()
            app.quality_var.set(keep_q)
            app._on_quality()
    check("пояснение в Настройках - одна строка для выбранного варианта", one_hint)

    def voices_open_selected():
        app._fx_voice = None
        app.show("book")
        app.show("voices")
        app.update()
        v = app._voice_obj()
        must(v is not None and app._fx_voice == v.id, f"справа {app._fx_voice}, выбран {v and v.id}")
        must(app.fx_title.cget("text") != "Выберите голос слева", "справа «Выберите голос слева»")
        app.show("book")
    check("«Голоса» открываются с голосом озвучки справа", voices_open_selected)

    def book_lock():
        lockp = HL.lock_path()
        must(str(lockp).startswith(TD.ROOT), f"проверка трогает настоящий замок: {lockp}")
        seen = []

        def fake_convert(path, **_k):
            seen.append(HL.owner_pid(lockp) == os.getpid() and HL.owner(lockp) == "Herald")
            return P.ConvertResult(None, None, 1, 1.0, "t", "a")

        def foreign():
            lockp.mkdir()
            (lockp / "owner").write_text(f"Понятный документ Fri Oct 23 10:00:00 2026 pid {os.getppid()}\n",
                                         encoding="utf-8")
        real = G.convert_book
        G.convert_book = fake_convert
        try:
            app._set_book(book)
            pump(lambda: app.est is not None)
            app.do_start()
            must(pump(lambda: not app.running, 10), "озвучка не закончилась")
            must(seen == [True], f"во время книги замок не у Herald: {seen}")
            must(not lockp.exists(), "замок не снят после книги")
            foreign()
            asked = []
            app._ask = lambda *a, **k: (asked.append(a), "wait")[1]
            seen.clear()
            app.do_start()
            must(asked and "Понятный документ" in str(asked[0]), f"не спросил: {asked}")
            must(pump(lambda: "Жду очереди" in app.status.cget("text"), 5), app.status.cget("text"))
            must(not seen, "озвучка пошла при чужом замке")
            _shutil.rmtree(lockp)
            must(pump(lambda: not app.running, 15), "после снятия чужого замка озвучка не прошла")
            must(seen == [True] and not lockp.exists(), f"очередь: {seen}, замок {lockp.exists()}")
            foreign()
            seen.clear()
            app.do_start()
            pump(lambda: "Жду очереди" in app.status.cget("text"), 5)
            app.do_stop()
            must(pump(lambda: not app.running, 10), "«Остановить» не отменил ожидание")
            must(not seen and lockp.exists() and "Понятный документ" in HL.owner(lockp),
                 "ожидание отменено, а чужой замок тронут или книга пошла")
            _shutil.rmtree(lockp)
            app._ask = lambda *a, **k: None                    # «Отмена» - ничего не начинается
            foreign()
            app.do_start()
            must(not app.running, "«Отмена» начала озвучку")
        finally:
            G.convert_book = real
            app.__dict__.pop("_ask", None)
            _shutil.rmtree(lockp, ignore_errors=True)
            app._check_resume()
    check("замок видеокарты: книга берёт и отдаёт, чужой - вопрос и очередь, «Остановить» в очереди",
          book_lock, limit=60)

    def close_asks():
        asked, gone = [], []
        app.destroy = lambda: gone.append(1)
        try:
            app._set_busy(True, "тест")
            app._ask = lambda *a, **k: (asked.append(a), None)[1]
            app.on_close()
            must(asked and "Остановить и закрыть" in str(asked[0]), f"вопроса нет: {asked}")
            must(not gone and app.running and not app._stop_flag.is_set(), "«Отмена» закрыла или остановила")
            app._ask = lambda *a, **k: "close"
            app.on_close()
            must(gone and app._stop_flag.is_set(), "«Остановить и закрыть» не остановило")
            app._set_busy(False)
            asked.clear()
            gone.clear()
            app._ask = lambda *a, **k: (asked.append(a), "close")[1]
            app.on_close()
            must(not asked and gone, "без озвучки окно спросило")
        finally:
            del app.destroy
            app.__dict__.pop("_ask", None)
            app._set_busy(False)
    check("закрытие во время озвучки: вопрос, «Отмена» - идёт дальше; без озвучки - без вопроса", close_asks)

    def ask_labels():
        """Своё окно вопроса: кнопки подписаны действием, Esc - «Отмена»."""
        texts = []

        def look():
            d = app._ask_dlg
            for w in d.winfo_children():
                for c in [w] + list(w.winfo_children()):
                    if isinstance(c, ctk.CTkButton):
                        texts.append(c.cget("text"))
            d.event_generate("<Escape>")
        app.after(300, look)
        got = app._ask("Идёт озвучка", "текст", [("close", "Остановить и закрыть", "danger"),
                                                 (None, "Отмена", "ghost")])
        must(texts == ["Остановить и закрыть", "Отмена"] and got is None, (texts, got))
    check("окно вопроса: кнопки «Остановить и закрыть» / «Отмена», Esc - отмена", ask_labels)

    def voice_changed():
        sd = os.path.join(P.output_dir_for(book), P.STATE_DIR)
        os.makedirs(sd, exist_ok=True)
        v = next(x for x in app._voices if x.id == TD.TRAINED)
        keep = app.voice_var.get()
        app.voice_var.set(v.label())
        app._set_book(book)
        pump(lambda: app.est is not None)
        engine, kw = app._eng_choice()
        fp = P.settings_fingerprint(engine, dict(kw, language=app.est.language), v.id, app._conv_options())
        open(os.path.join(sd, "part0001.flac"), "wb").close()
        st = {"fingerprint": fp, "total": 3, "finished": False, "parts": {"1": {"flac": "part0001.flac"}}}
        stf = os.path.join(sd, P.STATE_FILE)
        try:
            for mark, want in (("старые веса|1|1", True), (P.voice_weights_mark(v.id), False), (None, False)):
                d = dict(st) if mark is None else dict(st, voice_weights=mark)
                _json.dump(d, open(stf, "w"))
                app._check_resume()
                must(pump(lambda: "Продолжить" in app.start_btn.cget("text"), 5), app.start_btn.cget("text"))
                must(pump(lambda: bool(app.vchg_lbl.winfo_manager()) == want, 5),
                     f"отметка {mark!r}: строка «Голос изменился» {'нет' if want else 'лишняя'}")
        finally:
            P.forget_progress(book)
            app.voice_var.set(keep)
            app._check_resume()
            pump(lambda: not app.vchg_lbl.winfo_manager(), 5)
    check("«Продолжить» после смены весов голоса - «Голос изменился…», без смены - тихо", voice_changed)

    print("ошибки в фоне - понятным сообщением, без зависания (задание 13 А):")

    def bad_book():
        """Оценка прошлой книги ещё идёт (поток держит 6 с) - новая, нечитаемая, её не ждёт,
        а поздний ответ прошлой книги не затирает сообщение (задание 14 А)."""
        import threading as _th
        bad = os.path.join(TD.ROOT, "Книги", "битая.docx")
        open(bad, "wb").write(b"PK\x03\x04 not a docx" * 10)
        real_est = G.estimate_book
        release = _th.Event()

        def slow(path, minutes=15.0):
            if path == book:
                release.wait(6)
            return real_est(path, minutes)
        G.estimate_book = slow
        try:
            app._set_book(book)                    # прошлая книга: оценка висит
            app._set_book(bad)
            t0 = time.time()
            must(pump(lambda: "Не удалось прочитать книгу" in app.est_label.cget("text"), 5),
                 f"окно застряло: {app.est_label.cget('text')!r}")
            took = time.time() - t0
            release.set()
            pump(lambda: False, 1.0)               # поздний ответ прошлой книги
            must("Не удалось прочитать книгу" in app.est_label.cget("text") and app.est is None,
                 f"поздний ответ прошлой книги затёр сообщение: {app.est_label.cget('text')!r}")
            must(took < 3, f"новая оценка ждала прошлую: {took:.1f} с")
        finally:
            release.set()
            G.estimate_book = real_est
            app.clear_book()
    check("нечитаемая книга - «Не удалось прочитать книгу: …», не «Считаю объём…»", bad_book)

    def bad_sample():
        bad = os.path.join(TD.ROOT, "битый образец.mp3")
        open(bad, "wb").write(b"ID3 not audio" * 50)
        errs = []
        real_open, real_dlg, real_err = G.filedialog.askopenfilename, G.ctk.CTkInputDialog, _mb.showerror

        class Dlg:
            def __init__(self, *a, **k):
                pass

            def get_input(self):
                return "Битый образец"
        G.filedialog.askopenfilename = lambda *a, **k: bad
        G.ctk.CTkInputDialog = Dlg
        _mb.showerror = lambda *a, **k: errs.append(a)
        try:
            app.show("voices")
            app.load_sample()
            t0 = time.time()
            must(pump(lambda: bool(errs), 10), "ошибка не показана")
            must(time.time() - t0 < 5, f"битая запись: ошибка через {time.time() - t0:.0f} с (модель?)")
            must("не читается" in str(errs[0]), errs)
            must(not app.running, "окно осталось «занятым» после сбоя")
            must(app.add_voice_btn.cget("text") == "Добавить образец…", app.add_voice_btn.cget("text"))
            must("Не удалось подготовить голос" in app.vm_note.cget("text"), app.vm_note.cget("text"))
        finally:
            pump(lambda: not app.running, 30)     # не оставлять окно «занятым» следующим шагам
            G.filedialog.askopenfilename, G.ctk.CTkInputDialog, _mb.showerror = real_open, real_dlg, real_err
            app.show("book")
    check("«Добавить образец…» с битой записью - причина в окне сразу, без модели, кнопка снова доступна", bad_sample, limit=60)

    def bad_preview():
        errs = []
        real_err = _mb.showerror
        _mb.showerror = lambda *a, **k: errs.append(a)

        def broken(path, *, live=None, progress=None, **_k):
            raise RuntimeError("образец голоса не читается")
        G.preview_sample = broken
        try:
            app._set_book(book)
            pump(lambda: app.est is not None)
            app.do_preview()
            must(pump(lambda: bool(errs) and not app.running, 15), "пробник завис после сбоя")
            must("не читается" in str(errs[0]), errs)
        finally:
            _mb.showerror = real_err
            G.preview_sample = fake_preview
    check("«Проверить голос» с битым образцом - ошибка в окне, не зависание", bad_preview)

    def bad_read_text():
        errs = []
        real_err, keep_prep = _mb.showerror, P.prepare_read_text
        real_path = P.read_text_path
        _mb.showerror = lambda *a, **k: errs.append(a)

        def broken(path, opts):
            raise ValueError("ударения не расставились")
        P.prepare_read_text = broken
        P.read_text_path = lambda path, out_dir=None: "/нет/такого/файла"
        try:
            app._set_book(book)
            app.show_read_text()
            must(pump(lambda: bool(errs), 10), "ошибка не показана")
            must("не расставились" in str(errs[0]), errs)
            must(pump(lambda: app.readtext_btn.cget("text") == "Как будет прочитано", 5),
                 "кнопка не вернулась")
        finally:
            _mb.showerror, P.prepare_read_text, P.read_text_path = real_err, keep_prep, real_path
    check("«Как будет прочитано» со сбоем - сообщение с причиной, кнопка вернулась", bad_read_text)

    if DEFER is not None:
        import random
        i = sys.argv.index("--shuffle")
        seed = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else random.randrange(10 ** 6)
        jobs, DEFER = DEFER, None
        random.Random(seed).shuffle(jobs)
        print(f"случайный порядок, зерно {seed}: {' → '.join(n.split(':')[0][:18] for n, *_r in jobs)}")
        for name, fn, limit, skip in jobs:
            check(name, fn, limit, skip)

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
                if menu.type(i) not in ("cascade", "command"):
                    continue                           # «отрывная» линия меню X11 - без подписи
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
            # «Подробности» при EN (задание 14 Б3): строки окна и ядра - по-английски
            import io
            e.log.delete("1.0", "end")
            tee = G.App._StderrToLog(e, io.StringIO())
            tee.write("ℹ Продолжаю: готовых частей из прошлого раза - 1 из 2\n")
            tee.write("ℹ Ударения: у 3 слов вне словаря выровнено по книге (7 мест)\n")
            e._logln(G._("Остановлено. Готовые части остались на диске."))
            e._logln("сервис: модели выгружены после простоя")
            e._log_settings(G._("Озвучка книги"), "f5mlx", {}, "aidar") if e.book_path else None
            end = time.time() + 2
            while time.time() < end:
                e.update()
                time.sleep(0.03)
            body = e.log.get("1.0", "end")
            must(body.count("\n") >= 4 and not _re.search(r"[А-Яа-яЁё]", body), f"«Подробности» по-русски: {body!r}")
            must(abs(e._scale - (G.load_config().get("scale") or G.DEFAULT_SCALE)) < 1e-6
                 and G.DEFAULT_SCALE == 1.2, "масштаб по умолчанию не 120 %")
            must(G.VERSION in e.version_lbl.cget("text"), "версия внизу не совпадает с пакетом")
            must(G.VERSION == open("core/VERSION", encoding="utf-8").read().strip(), "версия не из core/VERSION")
            e.show("help")
            e.update()
            must(G.VERSION in e.help_box.get("1.0", "end"), "в Справке нет версии")
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

    def andrey_untouched():
        ch = TD.andrey_changed()
        must(not ch, f"изменились файлы Андрея: {ch[:10]}")
    check("данные Андрея (голоса, настройки, корзина) не изменились - хэши до и после", andrey_untouched)
    if SKIPS:
        print(f"пропущено с причиной: {len(SKIPS)}")
    print("всё в порядке" if not FAILS else f"\nПРОВАЛЕНО: {len(FAILS)}")
    for f in FAILS:
        print("   ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
