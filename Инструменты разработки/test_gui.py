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
    for title in tts_gui.SPEED_TITLES:
        check(f"темп «{title}»", lambda t=title: app.speed_var.set(t))
    check("параметры движка собираются", lambda: app._eng_kwargs())
    check("параметры вывода собираются", lambda: app._conv_options())

    print("копирование отчёта:")
    app._logln("проверочная строка")
    app.update()
    check("кнопка «Скопировать отчёт»", lambda: app.copy_log())
    check("в буфере то, что нужно",
          lambda: "проверочная" in app.clipboard_get() or (_ for _ in ()).throw(
              AssertionError(f"в буфере: {app.clipboard_get()[:40]!r}")))

    print("настройки:")
    check("окно настроек открывается", lambda: (app.show_settings(), app.update()))
    check("список голосов в настройках не пуст",
          lambda: app.voices_list.get("1.0", "end").strip() or (_ for _ in ()).throw(
              AssertionError("пусто")))
    for w in app.winfo_children():          # закрываем окно настроек
        if isinstance(w, ctk_toplevel):
            w.destroy()
    app.update()

    print("звучание голоса:")
    vid = app.current_voice()
    before = load_fx(default_voices_dir(), vid)
    check("окно звучания открывается", lambda: (app.show_voice_fx(), app.update()))
    check("правится ИМЕННО выбранный голос",
          lambda: vid == app.current_voice() or (_ for _ in ()).throw(
              AssertionError("окно открылось не для выбранного голоса")))

    def move_sliders():
        for key, (sl, _lbl, _u) in app._fx_vars.items():
            sl.set(-2 if key == "pitch" else 3)
            app._fx_changed(key)
        app.update()
    check("ползунки двигаются", move_sliders)
    check("значения читаются",
          lambda: app._fx_current().pitch == -2 or (_ for _ in ()).throw(
              AssertionError(f"вышло {app._fx_current()}")))
    check("сохранение не падает", lambda: save_fx(default_voices_dir(), vid,
                                                  app._fx_current()))
    check("сохранённое читается обратно",
          lambda: load_fx(default_voices_dir(), vid).pitch == -2 or (
              _ for _ in ()).throw(AssertionError("не сохранилось")))
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
