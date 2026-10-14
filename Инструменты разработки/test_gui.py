#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон окна без человека: открыть, потыкать, поймать ошибки (окно 3.0).

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
    check("три страницы построены заранее",
          lambda: must(set(app.pages) == {"book", "voices", "settings"}, app.pages.keys()))

    print("боковое меню и темы:")

    def pages_switch():
        for k in ("voices", "settings", "book"):
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

    def fx_open():
        app.select_fx_voice(vid)
        app.update()
        must(app._fx_voice == vid and app.fx_title.cget("text"), "не открылось")
        must(str(app.fx_play_btn.cget("state")) == "normal", "«Прослушать» выключена")
    check("звучание голоса открывается справа", fx_open)

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
        for key in profiles.PROFILES:
            app.profile_var.set(key)
            app._on_profile()
            app.update()
            engine, kw = app._eng_choice()
            ready = not profiles.get(key).clone
            must(all(v.ready == ready for v in app._voices), f"{key}: не те голоса")
            must(engine == ("silero" if ready else "f5mlx"), f"{key}: движок {engine}")
        app.profile_var.set("main")
        app._on_profile()
        e, kw = app._eng_choice()
        must(kw.get("steps") == 16 and "variant" not in kw, f"основной: {kw}")
    check("четыре движка в Настройках и нужные голоса", profiles_switch)
    app.profile_var.set(cfg_backup.get("profile", "main"))
    app._on_profile()

    print("подробности:")
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

    G.save_config(cfg_backup)
    app.destroy()
    print("всё в порядке" if not FAILS else f"\nПРОВАЛЕНО: {len(FAILS)}")
    for f in FAILS:
        print("   ", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
