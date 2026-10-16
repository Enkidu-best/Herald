#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Звук потоком (3.4) без моделей: заглушка движка с заданным RTF.

  * старт вовремя: при RTF 0,1 - сразу после первого куска; при 1,0 и 1,8 -
    позже, но без запинок до конца;
  * ни один кусок не пропал и не повторился: сыгранное = сырой звук потока =
    звук того же текста без потока, до сэмпла;
  * «Стоп» обрывает звук сразу и синтез - на ближайшем стыке;
  * пауза держит звук, синтез идёт дальше;
  * полировка по кускам почти не отличается от полировки целого файла.

    python "Инструменты разработки/test_stream.py"
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import threading
import time
import zlib

import numpy as np

fails: list[str] = []


def must(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        fails.append(msg)


SR = 24000
LETTERS_PER_SEC = 40.0          # «речь» заглушки втрое быстрее настоящей: тест короче


class StubEngine:
    """Движок-заглушка: звук по тексту (детерминированный), счёт = RTF × длина."""
    name = "stub"
    sample_rate = SR
    max_chunk_chars = 187
    stress_format = None
    supports_cloning = True

    def __init__(self, rtf: float):
        self.rtf = rtf
        self.calls: list[str] = []
        self.busy = threading.Event()

    def cache_key(self):
        return f"stub|{self.rtf}"

    def synth_batch(self, texts, voice, muls=None, hms=None):
        out = []
        for t in texts:
            self.calls.append(t)
            letters = sum(c.isalnum() for c in t)
            dur = max(0.3, letters / LETTERS_PER_SEC)
            self.busy.set()
            time.sleep(self.rtf * dur)
            self.busy.clear()
            n = int(dur * SR)
            f = 150 + zlib.crc32(t.encode()) % 300
            tone = 0.3 * np.sin(2 * np.pi * f * np.arange(n) / SR).astype(np.float32)
            out.append(np.concatenate([np.zeros(int(0.05 * SR), np.float32), tone,
                                       np.zeros(int(0.2 * SR), np.float32)]))
        self.last_cut_risk = [False] * len(texts)
        return out

    def synth_chunk(self, text, voice, **kw):
        return self.synth_batch([text], voice)[0]


TEXT = ("Море к вечеру стихло, и над водой повис туман. Лодки вернулись к берегу одна за "
        "другой. Старик долго смотрел на горизонт, потом медленно пошёл к дому.\n\n"
        "— Завтра будет ветер, — сказал он.\n\n— Откуда ты знаешь?\n\n"
        "Он не ответил. В доме было тихо и тепло, на столе остывал чай. Где-то далеко "
        "кричали чайки, и этот крик был похож на чей-то голос. Ночь пришла быстро.")


def chunks_for(text):
    import core.pipeline as P
    from core.stream import split_first
    return split_first(P._chunk_paragraphs(text, StubEngine.max_chunk_chars, 0.4, 0.8, 1.1))


def run_stream(rtf, *, gate=True, hd=False, prior=None, sink=None):
    import core.pipeline as P
    from core.stream import Live, Speed, ClockSink
    eng = StubEngine(rtf)
    ch = chunks_for(TEXT)
    sp = Speed("stub", prior if prior is not None else rtf, path=None)
    sp.spc = 1.0 / LETTERS_PER_SEC          # как будто уже мерили этот голос
    sink = sink or ClockSink(realtime=True)
    live = Live(sink, sp, hd=hd, gate=gate)
    t0 = time.monotonic()
    audio = P._synth_chunks(eng, "v", ch, SR, 0.4, lang="ru", use_cache=False,
                            sound_check=False, live=live)
    synth = time.monotonic() - t0
    live.wait(120)
    return eng, ch, live, sink, audio, synth


def main() -> int:
    import core.pipeline as P
    from core import stream as S
    P._asr_failed = True                    # без распознавания: заглушке не нужно

    print("Расчёт старта (без движка):")
    must(S.wait_needed(3.0, [(0.3, 3.0)] * 5) == 0.0, "RTF 0,1: играть сразу")
    w = S.wait_needed(3.0, [(5.4, 3.0)] * 5)
    must(w > 0, f"RTF 1,8: сначала подождать ({w:.1f} с)")
    must(S.wait_needed(3.0 + 5 * 3.0, []) == 0.0, "всё готово - играть")
    sp = S.Speed("x", 1.0, path=None)
    for d in (2, 4, 6, 8, 10, 12):          # от 6 замеров прикидка уже не нужна
        sp.observe(d * 12, d, 3 + 1.5 * d)
    a, b = sp._fit()
    must(abs(a - 3) < 0.01 and abs(b - 1.5) < 0.01, f"скорость: счёт = {a:.2f} + {b:.2f} × звук")

    print("Первый кусок короткий:")
    ch = chunks_for(TEXT)
    must(S.FIRST_MIN <= len(ch[0][0]) <= S.FIRST_MAX, f"первый кусок {len(ch[0][0])} знаков")
    joined = " ".join(c[0] for c in ch).replace("  ", " ")
    must(all(w in joined for w in ("туман.", "Лодки", "горизонт")), "текст не потерялся")

    ref = None
    for rtf in (0.1, 1.0, 1.8):
        print(f"RTF {rtf}:")
        eng, ch, live, sink, audio, synth = run_stream(rtf)
        if ref is None:
            ref = P._synth_chunks(StubEngine(0.0), "v", ch, SR, 0.4, lang="ru",
                                  use_cache=False, sound_check=False)
        played = np.concatenate(sink.chunks_out)
        must(np.array_equal(live.audio(), audio), "сырой звук потока = то, что вернул синтез")
        must(np.array_equal(audio, ref), "итоговый звук = тот же текст без потока")
        must(np.array_equal(played, audio), "сыграно всё, без пропусков и повторов")
        must(len(eng.calls) == len(ch), f"каждый кусок посчитан один раз ({len(eng.calls)})")
        dur = len(audio) / SR
        s1 = rtf * max(0.3, sum(c.isalnum() for c in ch[0][0]) / LETTERS_PER_SEC)
        fs = live.first_sound
        print(f"     звук {dur:.1f} с, счёт {synth:.1f} с, первый звук через {fs:.2f} с, "
              f"запинок {live.stalls} ({sink.stall_sec:.2f} с)")
        if rtf < 1:
            must(fs < s1 + 0.5, f"звук сразу после первого куска ({fs:.2f} с)")
        else:
            must(fs < synth * 0.9, f"звук раньше, чем досчитано всё ({fs:.1f} из {synth:.1f} с)")
        must(live.stalls == 0, "без запинок")

    print("Стоп:")
    import core.pipeline as P2
    from core.stream import Live, Speed, ClockSink
    eng = StubEngine(1.0)
    sink = ClockSink(realtime=True)
    live = Live(sink, Speed("stub", 1.0, path=None), hd=False, gate=False)
    res = {}

    def work():
        try:
            P2._synth_chunks(eng, "v", chunks_for(TEXT), SR, 0.4, lang="ru",
                             use_cache=False, sound_check=False, live=live)
            res["r"] = "досчитал"
        except P2.Cancelled:
            res["r"] = "стоп"
    th = threading.Thread(target=work)
    th.start()
    while live.first_sound is None and th.is_alive():
        time.sleep(0.05)
    time.sleep(0.5)
    t = time.monotonic()
    live.stop()
    pos = sink.position
    time.sleep(0.3)
    must(sink.position == pos, "звук оборвался сразу")
    th.join(10)
    took = time.monotonic() - t
    must(res.get("r") == "стоп" and took < 3.0,
         f"синтез бросил работу на стыке ({res.get('r')}, {took:.1f} с)")
    must(len(eng.calls) < len(chunks_for(TEXT)), "дальше не считал")

    print("Пауза:")
    eng = StubEngine(0.1)
    sink = ClockSink(realtime=True)
    live = Live(sink, Speed("stub", 0.1, path=None), hd=False)
    th = threading.Thread(target=lambda: P2._synth_chunks(
        eng, "v", chunks_for(TEXT), SR, 0.4, lang="ru", use_cache=False,
        sound_check=False, live=live))
    th.start()
    while live.first_sound is None:
        time.sleep(0.05)
    time.sleep(0.3)
    live.pause()
    p0 = sink.position
    time.sleep(0.6)
    must(sink.position == p0, "на паузе звук стоит")
    th.join(30)
    must(len(eng.calls) == len(chunks_for(TEXT)), "синтез на паузе идёт дальше")
    live.resume()
    must(live.wait(60) and sink.position > p0, "после паузы доиграл")
    must(sink.stalls == 0, "пауза - не запинка")

    print("Полировка по кускам против целого файла:")
    eng, ch, live, sink, audio, _s = run_stream(0.0, hd=True)
    played = np.concatenate(sink.chunks_out)
    from core.assemble import HD_FILTER, _ff
    import subprocess
    whole = np.frombuffer(subprocess.run(
        [_ff(), "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-af", HD_FILTER, "-f", "f32le", "-ar", "48000", "-ac", "1", "-"],
        input=audio.astype(np.float32).tobytes(), capture_output=True, check=True).stdout,
        dtype=np.float32)
    n = min(len(whole), len(played))
    must(abs(len(whole) - len(played)) < 48000 * 0.05,
         f"длина та же ({len(played) / 48000:.2f} против {len(whole) / 48000:.2f} с)")
    diff = whole[:n] - played[:n]
    snr = 10 * np.log10(np.sum(whole[:n] ** 2) / max(1e-12, np.sum(diff ** 2)))
    must(snr > 30, f"разница по кускам и целиком - на {snr:.0f} дБ тише звука")
    # щелчок на стыке - скачок между соседними сэмплами больше, чем внутри звука
    jumps = np.abs(np.diff(played))
    must(jumps.max() < np.abs(np.diff(whole)).max() * 1.5 + 1e-3, "стыки без щелчков")

    print("\nOK: звук потоком" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
