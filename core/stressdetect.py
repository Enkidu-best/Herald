"""Где в звуке ударение (3.8): какой слог модель сделала ударным.

Текст до модели верный («+о́тмель»), а клон иногда читает по-своему («отмЕль»).
Распознавание тут не судья - Whisper пишет слово по словарю. Ударный слог в
русском длиннее, громче и обычно выше по тону, поэтому для каждой гласной слова
берём её отрезок по выравниванию букв (core/align: пики CTC) и меряем:
  * dur - длительность гласной (от её пика до пика следующей буквы),
  * en  - наибольшая громкость на отрезке, дБ,
  * f0  - средняя высота тона, полутоны (если гласная звонкая).
Ударной считается гласная с наибольшей суммой отклонений от средних по слову.
Веса - по проверке на Silero, который «+» слушается строго (bench_stubborn.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOWELS = set("аеёиоуыэюя")
W_DUR, W_EN, W_F0 = 1.0, 1.0, 0.5


@dataclass
class Vowel:
    pos: int            # номер буквы в слове (norm)
    start: float
    end: float
    dur: float
    en: float
    f0: float           # полутоны от 100 Гц, nan - глухо


def _rms_db(a: np.ndarray, sr: int, hop: float = 0.005, win: float = 0.025):
    h, w = int(sr * hop), int(sr * win)
    if len(a) < w:
        return np.array([-80.0]), hop
    fr = np.lib.stride_tricks.sliding_window_view(a, w)[::h]
    return 20 * np.log10(np.sqrt((fr.astype(np.float64) ** 2).mean(axis=1)) + 1e-6), hop


def _f0(a: np.ndarray, sr: int):
    import librosa
    y = a if sr == 16000 else librosa.resample(a, orig_sr=sr, target_sr=16000)
    f0, vflag, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=640,
                                hop_length=80)
    st = 12 * np.log2(np.where(vflag & np.isfinite(f0), f0, np.nan) / 100.0)
    return st, 80 / 16000


def vowels(audio: np.ndarray, sr: int, ws, contours=None) -> list[Vowel]:
    """Гласные слова с мерками. ws - WordScore из core.align (со spans)."""
    sp = list(ws.spans)
    if not sp or len(sp) != len(ws.norm):
        return []
    en, hop = contours[0] if contours else _rms_db(audio, sr)
    f0, fhop = contours[1] if contours else _f0(audio, sr)
    out = []
    for j, (ch, t0, _t1) in enumerate(sp):
        if ch not in VOWELS:
            continue
        start = max(0.0, t0 - 0.02)
        if j + 1 < len(sp):
            end = sp[j + 1][1]
        else:                                   # гласная на конце слова - до спада громкости
            i0 = int(t0 / hop)
            seg = en[i0:i0 + int(0.35 / hop)]
            if len(seg):
                top = seg.max()
                k = np.argmax(seg < top - 12) if (seg < top - 12).any() else len(seg)
                end = t0 + k * hop
            else:
                end = t0 + 0.08
        end = max(end, start + 0.02)
        e = en[int(start / hop):max(int(end / hop), int(start / hop) + 1)]
        p = f0[int(start / fhop):max(int(end / fhop), int(start / fhop) + 1)]
        p = p[np.isfinite(p)]
        out.append(Vowel(j, start, end, end - start, float(e.max()) if len(e) else -80.0,
                         float(np.median(p)) if len(p) else float("nan")))
    return out


def scores(vs: list[Vowel], w=(W_DUR, W_EN, W_F0)) -> list[float]:
    """Насколько каждая гласная «ударная» (больше - вероятнее)."""
    if not vs:
        return []

    def z(x):
        x = np.asarray(x, float)
        ok = np.isfinite(x)
        if ok.sum() < 2:
            return np.zeros_like(x)
        m, s = x[ok].mean(), x[ok].std() + 1e-6
        return np.where(ok, (x - m) / s, 0.0)
    return list(w[0] * z([v.dur for v in vs]) + w[1] * z([v.en for v in vs])
                + w[2] * z([v.f0 for v in vs]))


def stressed(vs: list[Vowel], w=(W_DUR, W_EN, W_F0)) -> tuple[int, float]:
    """(номер ударной гласной среди гласных слова, отрыв от второй)."""
    s = scores(vs, w)
    if not s:
        return -1, 0.0
    o = np.argsort(s)[::-1]
    return int(o[0]), float(s[o[0]] - (s[o[1]] if len(o) > 1 else 0.0))


def vowel_index(stressed_word: str) -> int:
    """«к+аторжник» -> 0: номер ударной гласной среди гласных."""
    n = 0
    w = stressed_word.lower()
    for i, c in enumerate(w):
        if c == "+":
            return n
        if c in VOWELS:
            if c == "ё":
                return n
            n += 1
    return -1


# --- сравнение с образцами (основной способ) --------------------------------------
# Мерки гласных по пикам CTC оказались слабыми (на Silero 50-74 %): пик буквы стоит
# у её края, а не в середине. Надёжнее сравнить слово целиком с образцами того же
# слова с ударением на каждом слоге, озвученными ДРУГИМИ голосами (Silero «+»
# слушается строго). Сравниваются кадры выравнивателя (вероятности букв без
# «пусто»): они слышат «о» против «а», «е» против «и» и длину гласной почти
# независимо от голоса. Проверка на Silero (голос не входит в свои образцы):
# 420 из 456 = 92 % (89-96 % по голосам).

def segment(lp: np.ndarray, spans, blank: int = 0) -> np.ndarray:
    """Кадры слова: от первой буквы - 60 мс до последней + 120 мс, без «пусто»,
    логарифм нормированных вероятностей букв."""
    f0 = max(0, int((spans[0][1] - 0.06) / 0.02))
    f1 = int((spans[-1][2] + 0.12) / 0.02)
    p = np.exp(np.asarray(lp[f0:f1], np.float32))
    p = np.delete(p, blank, axis=1)
    p = p / (p.sum(1, keepdims=True) + 1e-6)
    return np.log(p + 1e-4)


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    C = 1 - an @ bn.T
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        row, prev = D[i], D[i - 1]
        for j in range(1, m + 1):
            row[j] = C[i - 1, j - 1] + min(prev[j - 1], prev[j], row[j - 1])
    return float(D[n, m] / (n + m))


def by_reference(seg: np.ndarray, refs: dict[int, list[np.ndarray]]) -> tuple[int, float]:
    """(номер ударной гласной, отрыв): ближайший по DTW набор образцов."""
    d = {k: float(np.mean([dtw(seg, r) for r in rs])) for k, rs in refs.items() if rs}
    if not d:
        return -1, 0.0
    o = sorted(d, key=d.get)
    return o[0], (d[o[1]] - d[o[0]]) if len(o) > 1 else 0.0
