#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Регресс: окончания слов в конце куска (Herald 3.1, раздел 2).

Фразы из приёмки, где раньше съедался последний звук: «бесконечных интриг»,
«я просто устал ждать», «ему в голову не выйдет». Каждая - отдельным куском и в
конце предложения, по нескольку зёрен случайности.

Диагноз 3.1: F5 растягивает речь на всё отведённое окно, и последняя согласная
упиралась в его границу (звук ещё шёл на краю); затухание ложилось прямо на
неё. Лечение - известная тишина в конце окна (f5_mlx.TAIL_ANCHOR_SEC).

Что проверяется у КАЖДОГО дубля прямо с выхода движка (без переозвучек):
  * сырой выход: у границы окна (перед тишиной-якорем) звука уже нет - тише
    -30 дБ к громкости речи; в 3.0 так было лишь в половине дублей;
  * хвост: после конца речи (Silero VAD) в куске остаётся не меньше 200 мс -
    звук после слова не срезан;
  * нет щелчка: последний отсчёт - ноль, последние 20 мс плавно затухают,
    в последних 100 мс нет скачков сигнала;
  * прежний срез по порогу громкости для справки: сколько он отрезал бы.
И по смыслу - Whisper large-v3-turbo (сильнее, чем small в самой программе,
чтобы оценка была независимой): последнее слово прозвучало ЦЕЛИКОМ, с
окончанием. Это проверяется у итога конвейера (с его переозвучками) - так,
как кусок попадёт в книгу; доля удачных сырых дублей печатается для справки.

    python "Инструменты разработки/test_endings.py" [--seeds 3]
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import argparse
import os
import contextlib
import io
import tempfile

import numpy as np

PHRASES = ["бесконечных интриг", "я просто устал ждать", "ему в голову не выйдет"]
CONTEXT = {
    "бесконечных интриг": "Он давно устал от дворцовых сплетен и бесконечных интриг",
    "я просто устал ждать": "Не сердись на меня, я просто устал ждать",
    "ему в голову не выйдет": "Такая мысль никогда ему в голову не выйдет",
}
VOICE = os.environ.get("HERALD_TEST_VOICE", "Вячеслав Герасимов")
JUDGE = "mlx-community/whisper-large-v3-turbo"
SR = 24000


def heard_words(audio: np.ndarray) -> list[str]:
    """Судья - Whisper large-v3-turbo (core/asr: MLX на Mac, faster-whisper иначе)."""
    import soundfile as sf
    from core import asr
    from core.pipeline import _words
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SR)
    try:
        r = asr.transcribe(f.name, "turbo", language="ru")
    finally:
        os.unlink(f.name)
    return _words(r["text"])


def seed(n: int) -> None:
    """Зерно случайности для движка этого компьютера (MLX или torch)."""
    try:
        import mlx.core as mx
        mx.random.seed(n)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(n)
    except ImportError:
        pass


def old_rms_end(wave: np.ndarray) -> int:
    """Прежний срез (до доработок): последний кадр громче 0,004 + 40 мс."""
    win = 240
    last = 0
    for i in range(len(wave) // win):
        if np.sqrt(np.mean(wave[i * win:(i + 1) * win] ** 2)) > 0.004:
            last = i
    return min(len(wave), (last + 1) * win + win * 4)


def click_problems(a: np.ndarray) -> list[str]:
    out = []
    if abs(float(a[-1])) > 1e-3:
        out.append(f"последний отсчёт {a[-1]:+.4f} (ступенька в паузу)")
    k = int(0.02 * SR)
    env = np.abs(a[-k:])
    # затухание: огибающая по 2 мс не растёт к концу
    blocks = env[: k // 48 * 48].reshape(-1, 48).max(axis=1)
    if len(blocks) > 2 and blocks[-1] > blocks[0] * 1.05 + 1e-4:
        out.append("последние 20 мс не затухают")
    tail = a[-int(0.1 * SR):]
    d = np.abs(np.diff(tail))
    body = np.abs(np.diff(a[: max(1, len(a) - len(tail))]))
    ref = float(np.percentile(body, 99.9)) if len(body) else 0.05
    if len(d) and float(d.max()) > max(0.05, 1.5 * ref):
        out.append(f"скачок {d.max():.3f} в последних 100 мс")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    import core.engines.f5_mlx as F
    from core.engines import get_engine
    from core.pipeline import _prepare_text, _synth_chunks, _ends_like, _with_end_punct, _words
    from core.vad import speech_end

    from core import profiles
    from core.engines import clone_engine
    eng = get_engine(clone_engine(), language="ru", **profiles.get(None).kwargs)
    eng.ensure_loaded()
    raw_store = []
    orig = F._finish_tail

    def spy(w):
        raw_store.append(np.array(w, copy=True))
        return orig(w)
    F._finish_tail = spy

    fails, raw_ok, raw_n, old_cut = [], 0, 0, []
    cases = [(p, p) for p in PHRASES] + [(p, CONTEXT[p]) for p in PHRASES]
    for phrase, text in cases:
        stressed = _with_end_punct(_prepare_text(text, "+", True, True, "ru"))
        want = _words(phrase)[-1]
        for sd in range(a.seeds):
            seed(100 + sd)
            raw_store.clear()
            audio = eng.synth_chunk(stressed, VOICE)
            raw = raw_store[-1]
            tag = f"«{text}» зерно {sd}"
            se = speech_end(audio, SR)
            if se is None:
                fails.append(f"{tag}: речи не найдено")
                continue
            tail = len(audio) / SR - se
            if tail < 0.2 - 1e-3:
                fails.append(f"{tag}: после речи всего {tail * 1000:.0f} мс")
            for p in click_problems(audio):
                fails.append(f"{tag}: щелчок - {p}")
            w = 240
            seg = raw[:len(raw) - int(F.TAIL_ANCHOR_SEC * SR)]
            e = np.sqrt(np.mean(seg[: len(seg) // w * w].reshape(-1, w) ** 2, axis=1))
            edge = 20 * np.log10(e[-3:-1].mean() / (np.percentile(e, 95) + 1e-9) + 1e-9)
            if edge > -30:
                fails.append(f"{tag}: слово упирается в границу окна ({edge:.0f} дБ)")
            raw_se = speech_end(raw, SR)
            if raw_se is not None:
                old_cut.append(raw_se - old_rms_end(raw) / SR)
            raw_n += 1
            got = heard_words(audio)
            ok = _ends_like(want, got)
            raw_ok += ok
            # итог конвейера: та же фраза так, как она попадёт в книгу
            seed(100 + sd)
            final = _synth_chunks(eng, VOICE, [(text, 0.3, 1.0)], SR, 0.3, use_cache=False)
            fgot = heard_words(final)
            fok = _ends_like(want, fgot)
            print(f"  {tag}: хвост {tail * 1000:.0f} мс, сырой дубль "
                  f"{'целиком' if ok else 'НЕ целиком'} ({' '.join(got[-3:])}), "
                  f"итог {'целиком' if fok else 'НЕ ЦЕЛИКОМ'} ({' '.join(fgot[-3:])})")
            if not fok:
                # что слышит на этом месте акустическая модель (без языковой
                # догадки) - судьи бывают несогласны (3.5: «выйди» / «выйдет»)
                try:
                    from core import align
                    lw = align.aligner().score(final, SR, _with_end_punct(text).replace("+", ""))[-1]
                    note = f" (выравнивание слышит «{lw.heard}», уверенность {lw.post_mean:.2f})"
                except Exception:
                    note = ""
                fails.append(f"{tag}: в итоге последнее слово не целиком: {' '.join(fgot[-3:])}{note}")
            # стык «хвост куска -> вставленная пауза»: до последнего ненулевого отсчёта
            nz = np.nonzero(final)[0]
            for p in click_problems(final[: nz[-1] + 2] if len(nz) else final):
                fails.append(f"{tag}: щелчок в итоге - {p}")
    F._finish_tail = orig

    print(f"\nСырые дубли (без переозвучек): последнее слово целиком {raw_ok} из {raw_n}")
    if old_cut:
        late = [c for c in old_cut if c > 0.0]
        print(f"Прежний срез по громкости отрезал бы речь по VAD в {len(late)} из "
              f"{len(old_cut)} дублей (до {max(old_cut) * 1000:.0f} мс)")
    if fails:
        print(f"\nПРОВАЛ ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return 1
    print("\nOK: последний звук на месте, хвост ≥ 200 мс, щелчков нет")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
