#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон окна без человека: открыть, потыкать, поймать ошибки.

Родился из истории, где одну и ту же кнопку ломали трижды подряд, а замечал
это только пользователь. Тест поднимает настоящее окно, дёргает обработчики
и ловит всё, что упало, — включая исключения из tkinter-колбэков, которые
иначе просто печатаются в консоль и никого не останавливают.

    python "Инструменты разработки/test_gui.py"

Синтез здесь НЕ запускается: проверяется интерфейс, а не звук.
"""

from __future__ import annotations

import os as _os, sys as _sys
# скрипт лежит в подпапке, а core/ — в корне проекта
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import os
import traceback

os.environ.setdefault("PYTHONHASHSEED", "0")

FAILS: list[str] = []


def check(name: str, fn, limit: int = 20):
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


def main() -> int:
    import tkinter
    import customtkinter
    import tts_gui
    ctk_toplevel = customtkinter.CTkToplevel
    from core.voicefx import VoiceFX, load as load_fx, save as save_fx
    from core.engines.f5 import default_voices_dir

    # исключения внутри колбэков tkinter не роняют программу — перехватываем
    def report(self, exc, val, tb):
        FAILS.append(f"колбэк tkinter: {val!r}")
        print(f"  ПАДАЕТ колбэк tkinter: {val!r}")
        traceback.print_exception(exc, val, tb)
    tkinter.Tk.report_callback_exception = report

    print("окно:")
    app = tts_gui.App()
    app.update()

    # Страховка от зависания: тест не должен ждать вечно ни на одном шаге.
    import signal

    def timeout(_sig, _frm):
        raise TimeoutError("шаг теста не уложился в отведённое время")
    signal.signal(signal.SIGALRM, timeout)

    def wait(cond, seconds=8.0):
        """Покрутить НАСТОЯЩИЙ цикл событий, пока фоновый поток не отработает.

        Одного app.update() мало: фоновые потоки зовут self.after, а он без
        работающего mainloop падает с «main thread is not in main loop» —
        ровно так же, как падало бы в настоящем приложении, запусти мы его
        неправильно. Поэтому крутим mainloop и выходим по условию.
        """
        import time
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
    check("список голосов заполнен",
          lambda: app.voice_menu.cget("values") or (_ for _ in ()).throw(
              AssertionError("список голосов пуст")))

    print("книга:")
    book = "Примеры текстов/Пример — проба пера.txt"
    check("выбор книги", lambda: app._set_book(book))
    wait(lambda: app.est is not None)        # оценка считается в фоне
    check("оценка посчиталась",
          lambda: app.est or (_ for _ in ()).throw(AssertionError("оценки нет")))
    check("язык книги определён",
          lambda: app.est.language in ("ru", "en") or (_ for _ in ()).throw(
              AssertionError(f"язык: {app.est.language}")))
    check("очистка книги", lambda: (app.clear_book(), app.update()))

    print("переключение настроек:")
    check("смена движка на системный",
          lambda: (app.on_engine(tts_gui.ENGINE_SHORT["apple"]), app.update()))
    check("возврат на клон",
          lambda: (app.on_engine(tts_gui.ENGINE_SHORT["f5mlx"]), app.update()))
    for title in tts_gui.MODE_TITLES:
        check(f"качество «{title}»",
              lambda t=title: (app.mode_var.set(t), app.show_estimate(), app.update()))
    check("параметры движка собираются", lambda: app._eng_kwargs())
    check("параметры вывода собираются", lambda: app._conv_options())

    print("копирование отчёта:")
    app._logln("проверочная строка")
    app.update()
    check("кнопка «Скопировать отчёт»", lambda: app.copy_log())
    check("в буфере то, что нужно",
          lambda: "проверочная" in app.clipboard_get() or (_ for _ in ()).throw(
              AssertionError(f"в буфере: {app.clipboard_get()[:40]!r}")))

    def copy_selection():
        inner = app.log._textbox
        inner.tag_remove("sel", "1.0", "end")
        inner.tag_add("sel", "1.0", "1.11")
        app.copy_log(selection=True)
        got = app.clipboard_get()
        if "проверочн" not in got:
            raise AssertionError(f"выделенное не скопировалось: {got[:30]!r}")
    check("копируется выделенный фрагмент", copy_selection)
    check("привязки Cmd+C стоят на внутреннем виджете",
          lambda: app.log._textbox.bind("<Command-c>") or (_ for _ in ()).throw(
              AssertionError("привязки нет")))
    check("правая кнопка открывает меню",
          lambda: app.log._textbox.bind("<Button-3>") or (_ for _ in ()).throw(
              AssertionError("нет меню по правой кнопке")))

    print("отчёт:")
    app._set_book(book)
    wait(lambda: app.est is not None)
    check("шапка отчёта пишется", lambda: (app.log.delete("1.0", "end"),
                                           app._log_settings("Проверка"),
                                           app.update()))
    check("в шапке есть настройки голоса",
          lambda: "настройки голоса" in app.log.get("1.0", "end") or (
              _ for _ in ()).throw(AssertionError("шапка без настроек")))

    print("настройки:")
    check("окно настроек открывается", lambda: (app.show_settings(), app.update()))
    check("список голосов в настройках не пуст",
          lambda: app.voices_list.get("1.0", "end").strip() or (_ for _ in ()).throw(
              AssertionError("пусто")))
    for w in app.winfo_children():          # закрываем окно настроек
        if isinstance(w, ctk_toplevel):
            w.destroy()
    app.update()

    print("кнопка звучания:")

    def fx_button_stable():
        sizes = set()
        for label in app.voice_menu.cget("values"):
            app.voice_var.set(label)
            app.update_idletasks()
            if not app.fx_btn.winfo_ismapped():
                raise AssertionError(f"кнопка ♪ пропала при голосе «{label}»")
            sizes.add((app.fx_btn.winfo_width(), app.fx_btn.winfo_height()))
        if len(sizes) > 1:
            raise AssertionError(f"размер кнопки ♪ скачет: {sizes}")
    check("♪ одного размера при любом голосе", fx_button_stable)

    def layout_stable():
        """Поля настроек не должны прыгать при переключении голоса."""
        geoms = set()
        for label in app.voice_menu.cget("values"):
            app.voice_var.set(label)
            app.update_idletasks()
            geoms.add(tuple(w.winfo_x() for w in app._setting_menus))
        if len(geoms) > 1:
            raise AssertionError(f"поля сдвигаются: {len(geoms)} разных раскладок")
    check("поля не сдвигаются при смене голоса", layout_stable)
    check("длинное имя голоса укорачивается",
          lambda: len(app._voice_label(type("V", (), {"id": "О" * 40,
                                                      "language": "ru"})())) < 40
          or (_ for _ in ()).throw(AssertionError("имя не обрезается")))

    print("блокировка на время работы:")

    def busy_locks():
        app._set_busy(True, "тест")
        locked = [b for b in (app.start_btn, app.preview_btn, app.clear_btn,
                              app.fx_btn) if b.cget("state") != "disabled"]
        stop_live = app.stop_btn.cget("state") == "normal"
        app._set_busy(False)
        if locked:
            raise AssertionError(f"не заблокировано кнопок: {len(locked)}")
        if not stop_live:
            raise AssertionError("«Стоп» недоступна во время работы")
    check("во время работы активна только «Стоп»", busy_locks)

    print("пауза, проценты, таймер, длина файла:")

    def pause_cycle():
        import threading, time as _t
        app._set_busy(True, "тест")
        app.pause_btn.configure(state="normal")
        app.do_pause()
        if not app._pause_flag.is_set() or app.pause_btn.cget("text") != "Продолжить":
            raise AssertionError("пауза не встала")
        passed = threading.Event()
        th = threading.Thread(target=lambda: (app._pause_gate(), passed.set()))
        th.start()
        _t.sleep(0.8)
        if passed.is_set():
            raise AssertionError("синтез не ждёт на паузе")
        app.do_pause()                                   # продолжить
        th.join(2)
        app._set_busy(False)
        if not passed.is_set():
            raise AssertionError("после «Продолжить» синтез не пошёл")
    check("«Пауза» останавливает и «Продолжить» возобновляет", pause_cycle)

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
        if alive:
            raise AssertionError("«Стоп» не снимает паузу — синтез завис бы")
    check("«Стоп» работает и на паузе", stop_breaks_pause)

    def percent_shown():
        app._progress(0.42, "Озвучиваю: часть 2 из 5")
        app.update()
        if not app.status.cget("text").startswith("42%"):
            raise AssertionError(f"нет процентов: {app.status.cget('text')!r}")
    check("в статусе есть проценты", percent_shown)

    def part_shown():
        app._part_progress(2, 5, 0.5)
        app.update()
        if app.part_label.cget("text") != "Часть 2 из 5 — 50%":
            raise AssertionError(app.part_label.cget("text"))
    check("прогресс текущей части", part_shown)

    def clock_fmt():
        import tts_gui as G
        for sec, want in ((187, "3 мин 07 с"), (3912, "1 ч 05 мин 12 с")):
            if G._fmt_clock(sec) != want:
                raise AssertionError(f"{sec} с -> {G._fmt_clock(sec)!r}")
    check("таймер в часах и минутах", clock_fmt)

    def single_file_opt():
        import tts_gui as G
        app.len_var.set("60 минут")
        if app._conv_options().minutes_per_file != 60 or app._conv_options().single_file:
            raise AssertionError("60 минут не работает")
        app.len_var.set(G.SINGLE_FILE)
        o = app._conv_options()
        app.len_var.set("15 минут")
        if not o.single_file:
            raise AssertionError("«Один файл» не передаётся в синтез")
    check("длина 60 минут и «Один файл»", single_file_opt)

    print("звучание голоса:")
    vid = app.current_voice()
    before = load_fx(default_voices_dir(), vid)
    check("окно звучания открывается", lambda: (app.show_voice_fx(), app.update()))
    check("правится ИМЕННО выбранный голос",
          lambda: vid == app.current_voice() or (_ for _ in ()).throw(
              AssertionError("окно открылось не для выбранного голоса")))

    check("темп есть среди настроек голоса",
          lambda: "speed" in app._fx_vars or (_ for _ in ()).throw(
              AssertionError("ползунка темпа нет")))

    def move_sliders():
        for key, (sl, _lbl, _u) in app._fx_vars.items():
            sl.set(1.1 if key == "speed" else (-2 if key == "pitch" else 3))
            app._fx_slider(key)
        app.update()
    check("ползунки двигаются", move_sliders)
    check("значения читаются",
          lambda: (app._fx_current().pitch == -2
                   and abs(app._fx_current().speed - 1.1) < 0.03) or (
              _ for _ in ()).throw(AssertionError(f"вышло {app._fx_current()}")))
    check("сохранение не падает", lambda: save_fx(default_voices_dir(), vid,
                                                  app._fx_current()))
    check("сохранённое читается обратно",
          lambda: load_fx(default_voices_dir(), vid).pitch == -2 or (
              _ for _ in ()).throw(AssertionError("не сохранилось")))
    check("темп голоса сохранился",
          lambda: abs(load_fx(default_voices_dir(), vid).speed - 1.1) < 0.03 or (
              _ for _ in ()).throw(AssertionError("темп не сохранился")))
    check("поправки попадают в фильтр",
          lambda: "asetrate" in load_fx(default_voices_dir(), vid).filter_chain()
          or (_ for _ in ()).throw(AssertionError("фильтр пустой")))
    check("кнопка воспроизведения переживает закрытие окна",
          lambda: app._fx_button_ready())
    for w in app.winfo_children():
        if isinstance(w, ctk_toplevel):
            w.destroy()
    app.update()
    save_fx(default_voices_dir(), vid, before)      # возвращаем как было

    app.update()
    app.destroy()

    print()
    if FAILS:
        print(f"ПРОВАЛЕНО {len(FAILS)}:")
        for f in FAILS:
            print("   ", f)
        return 1
    print("всё в порядке")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
