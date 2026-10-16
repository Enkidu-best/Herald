#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Время первого звука (3.4, п. 6): Silero, клон на фразе, пробник, /speak.

Звук «играет» по часам (ClockSink) - тот же расчёт старта и учёт запинок, что
у звуковой карты, только без звука в колонках. Модель загружается заранее:
цели заданы для загруженной модели. Кэш синтеза выключен - иначе повтор
мгновенный.

    python "Инструменты разработки/bench_live.py" --case all --out live.json
    python "Инструменты разработки/bench_live.py" --case phrase --steps 8
    python "Инструменты разработки/bench_live.py" --case pair --out-dir пара/
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import json
import time

SILERO_PHRASE = "Дом стоял на краю деревни, и каждое утро над ним поднимался дым."
CLONE_PHRASE = ("Море к вечеру стихло, и над водой повис густой туман. Лодки одна за другой "
                "возвращались к берегу, а старик всё смотрел на горизонт.")
BOOK = "Книги/Жажда власти 1 - Сергей Тармашев/Жажда власти. Книга 1.fb2"


def first_sound(text, engine, kw, voice, prior, repeat, sound_check=False):
    from core import stream as S
    from core.engines import accel_label
    out = []
    for _ in range(repeat):
        live = S.Live(S.ClockSink(realtime=True),
                      S.make_speed(engine, kw, voice, sound_check, prior, accel_label()))
        audio, sr = S.speak(text, engine=engine, voice=voice, live=live, eng_kwargs=kw,
                            use_cache=False, sound_check=sound_check)
        synth = time.monotonic() - live.t0
        live.wait(600)
        out.append({"первый_звук_с": round(live.first_sound, 2), "счёт_с": round(synth, 2),
                    "звук_с": round(len(audio) / sr, 2), "запинок": live.stalls,
                    "запинки_с": round(live.sink.stall_sec, 2)})
        print("   ", out[-1], flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all", choices=["all", "silero", "phrase", "preview",
                                                       "server", "pair"])
    ap.add_argument("--voice", default="Вячеслав Герасимов")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--book", default=BOOK)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--no-old", action="store_true", help="пробник: без замера «как в 3.3»")
    ap.add_argument("--no-check", action="store_true", help="пробник без проверки слов по звуку")
    a = ap.parse_args()
    import core.pipeline as P
    from core import profiles, stream as S
    from core.engines import clone_engine, accel_label
    res = {"steps": a.steps}
    clone = clone_engine()
    kw = dict(profiles.get(None).kwargs, steps=a.steps)
    cases = ["silero", "phrase", "preview", "server"] if a.case == "all" else [a.case]

    if "silero" in cases:
        print("Silero, фраза (модель и ударения загружены):", flush=True)
        S.speak("Проверка.", engine="silero", voice="aidar", live=None)       # прогрев
        res["silero"] = first_sound(SILERO_PHRASE, "silero", {}, "aidar",
                                    profiles.SILERO.rtf, a.repeat)

    if "phrase" in cases:
        print(f"Клон, фраза ~10 с, {a.steps} шагов (модель загружена):", flush=True)
        S.speak("Проверка связи.", engine=clone, voice=a.voice, live=None, eng_kwargs=kw)
        res["phrase"] = first_sound(CLONE_PHRASE, clone, kw, a.voice,
                                    profiles.get(None).rtf, a.repeat)

    if "preview" in cases:
        print("Пробник 1 мин, как в окне (проверка слов по звуку включена):", flush=True)
        opts = P.ConvertOptions(use_cache=False, sound_check=not a.no_check)
        old = 0.0
        if not a.no_old:
            t0 = time.monotonic()
            P.preview_sample(a.book, engine=clone, eng_kwargs=dict(kw), voice=a.voice,
                             minutes=1.0, options=opts)
            old = time.monotonic() - t0
            print(f"    без потока (как в 3.3): весь пробник через {old:.1f} с", flush=True)
        else:
            S.speak("Проверка связи.", engine=clone, voice=a.voice, live=None, eng_kwargs=kw)
        live = S.Live(S.ClockSink(realtime=True),
                      S.make_speed(clone, dict(kw, language="ru"), a.voice, not a.no_check,
                                   profiles.get(None).rtf * (1.2 if not a.no_check else 1.0),
                                   accel_label()))
        t0 = time.monotonic()
        wav = P.preview_sample(a.book, engine=clone, eng_kwargs=dict(kw), voice=a.voice,
                               minutes=1.0, options=opts, live=live)
        synth = time.monotonic() - t0
        live.wait(900)
        import soundfile as sf
        info = sf.info(wav)
        print("    ход: (время выхода куска, его звук с паузой, готово к тому моменту):",
              live.timeline, flush=True)
        res["preview"] = {"без_потока_с": round(old, 1), "первый_звук_с": round(live.first_sound, 1),
                          "счёт_с": round(synth, 1), "звук_с": round(info.duration, 1),
                          "запинок": live.stalls, "запинки_с": round(live.sink.stall_sec, 2)}
        print("   ", res["preview"], flush=True)

    if "server" in cases:
        print("/speak, Silero, от запроса до первого звука в ответе:", flush=True)
        import socket
        import urllib.request
        from core.speak_server import SpeakServer
        sk = socket.socket()
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
        sk.close()
        srv = SpeakServer(port=port)
        srv._warm = lambda: None
        srv.start()
        SpeakServer._warm(srv)                       # модель и ударения - до замера
        out = []
        for _ in range(a.repeat):
            body = json.dumps({"text": SILERO_PHRASE, "fast": True}).encode()
            req = urllib.request.Request(f"{srv.url}/speak", data=body,
                                         headers={"Content-Type": "application/json"})
            t0 = time.monotonic()
            r = urllib.request.urlopen(req, timeout=60)
            r.read(44 + 2)                           # заголовок WAV и первый сэмпл
            t1 = time.monotonic() - t0
            rest = r.read()
            out.append({"первый_звук_с": round(t1, 3), "звук_с": round(len(rest) / 2 / 48000, 2)})
            print("   ", out[-1], flush=True)
        srv.stop()
        res["server"] = out

    if "pair" in cases:
        # «живой» профиль 8 шагов против 12 - для слепого сравнения на одной фразе
        import mlx.core as mx
        import soundfile as sf
        os_ = _os
        os_.makedirs(a.out_dir, exist_ok=True)
        for steps in (12, 8):
            k = dict(profiles.get(None).kwargs, steps=steps)
            S.speak("Проверка связи.", engine=clone, voice=a.voice, live=None, eng_kwargs=k)
            mx.random.seed(a.seed)
            audio, sr = S.speak(CLONE_PHRASE, engine=clone, voice=a.voice, live=None,
                                eng_kwargs=k, use_cache=False)
            path = os_.path.join(a.out_dir, f"фраза_{steps}.wav")
            sf.write(path, audio, sr)
            print("   ", path, flush=True)

    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
