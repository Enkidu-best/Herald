#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б: замер для контрольных точек - похожесть, темп по третям предложения,
слова, чистота (DNSMOS). Один и тот же набор чисел для всех вариантов.

  * похожесть - ECAPA каждого предложения к среднему вектору настоящих фрагментов чтеца
    (как в 11а, часть В), медиана;
  * темп по третям - слова Whisper с метками времени, предложение делится на три части по
    числу слов; слогов в секунду в начале / середине / конце; «разброс» - медиана по
    предложениям (макс - мин) / среднее трёх третей. «Желе - пулемёт» = низкое начало и
    высокая середина, большой разброс;
  * слова - доля слов текста, найденных в распознанном (числа Whisper пишет цифрами -
    одинаково для всех);
  * интонация (11б, п. 5) - размах тона в полутонах по тем же третям предложения;
  * DNSMOS (Microsoft, через speechmos): OVRL / SIG / BAK по окнам 10 с, среднее.
Живой чтец - его фразы из набора для обучения (текст сверен со звуком в 11а, Д).

    python "Инструменты разработки/measure_11b.py" <имя набора> <чтец> <файл>[=подпись] ...
    python "Инструменты разработки/measure_11b.py" живой "Игорь Ященко"
Пишет «Архив/11б/замеры.jsonl» (по строке на файл).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import difflib
import glob
import importlib.util
import json
import re

import numpy as np

BANK = "Архив/голоса диктора 3.9"
OUT = "Архив/11б"
VOW = "аеёиоуыэюя"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(w: str) -> str:
    return re.sub(r"[^а-яё]", "", w.lower()).replace("ё", "е")


def syl(w: str) -> int:
    return sum(c in VOW for c in w.lower())


def heard(P, a16) -> list[tuple[str, float, float]]:
    out, step = [], 30 * 16000
    for s in range(0, len(a16), step):
        for w, t0, t1 in P._listen(a16[s:s + step + 16000], 16000, "ru", P._ASR_FULL) or []:
            t0, t1 = t0 + s / 16000, t1 + s / 16000
            if out and t0 < out[-1][2] - 0.05:
                continue
            out.append((w, t0, t1))
    return out


def sentences_with_words(text: str, hw):
    """Предложение текста -> список услышанных слов (слово, начало, конец) по совпадению."""
    from razdel import sentenize
    sents = [s.text for s in sentenize(text)]
    sw = []
    for si, s in enumerate(sents):
        sw += [(norm(w), si) for w in re.findall(r"[А-Яа-яЁё]+", s.replace("+", ""))]
    hn = [norm(w) for w, _a, _b in hw]
    sm = difflib.SequenceMatcher(None, [w for w, _ in sw], hn, autojunk=False)
    per: dict[int, list] = {}
    for b in sm.get_matching_blocks():
        for i, j in zip(range(b.a, b.a + b.size), range(b.b, b.b + b.size)):
            per.setdefault(sw[i][1], []).append(hw[j])
    matched = sum(b.size for b in sm.get_matching_blocks())
    return sents, per, matched, len(sw)


def thirds(words) -> list[float] | None:
    """Слогов в секунду в трёх частях предложения (по числу слов); None - коротко."""
    if len(words) < 6:
        return None
    n = len(words)
    cuts = [0, n // 3, 2 * n // 3, n]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        part = words[a:b]
        dur = part[-1][2] - part[0][1]
        if dur <= 0.2:
            return None
        out.append(sum(syl(w) for w, _s, _e in part) / dur)
    return out


def pitch_thirds(a16, words) -> list[float] | None:
    """11б, п. 5: размах тона (полутонов, 90-й / 10-й перцентиль F0) в трёх частях
    предложения - «на одной ноте, как молитва» = узкий размах в середине и конце."""
    import librosa
    if len(words) < 6:
        return None
    n = len(words)
    cuts = [0, n // 3, 2 * n // 3, n]
    out = []
    for a, b in zip(cuts, cuts[1:]):
        t0, t1 = words[a][1], words[b - 1][2]
        seg = a16[int(t0 * 16000):int(t1 * 16000)]
        if len(seg) < 16000 * 0.4:
            return None
        f0, vf, _p = librosa.pyin(seg, fmin=60, fmax=400, sr=16000, frame_length=1024)
        f0 = f0[vf & np.isfinite(f0)]
        if len(f0) < 8:
            return None
        out.append(float(12 * np.log2(np.percentile(f0, 90) / np.percentile(f0, 10))))
    return out


def pitch_summary(pt: list[list[float]]) -> dict:
    if not pt:
        return {}
    t = np.array(pt)
    return {"размах тона, полутонов: начало / середина / конец": [round(float(x), 2) for x in np.median(t, axis=0)],
            "размах тона за предложение (среднее третей)": round(float(np.median(t.mean(axis=1))), 2)}


def dnsmos(a16) -> dict:
    from speechmos import dnsmos as D
    vals = []
    for s in range(0, max(1, len(a16) - 16000 * 3), 16000 * 10):
        w = a16[s:s + 16000 * 10]
        w = w / max(1.0, float(np.abs(w).max()))            # DNSMOS принимает только [-1, 1]
        if len(w) >= 16000 * 3:
            vals.append(D.run(w.astype(np.float32), sr=16000))
    if not vals:
        return {}
    return {k: round(float(np.mean([v[k] for v in vals])), 2) for k in ("ovrl_mos", "sig_mos", "bak_mos")}


def summarize(th: list[list[float]]) -> dict:
    if not th:
        return {}
    t = np.array(th)
    spread = np.median((t.max(axis=1) - t.min(axis=1)) / t.mean(axis=1))
    med = np.median(t, axis=0)
    return {"темп: начало / середина / конец": [round(float(x), 2) for x in med],
            "разброс темпа внутри предложения": round(float(spread), 3),
            "середина / начало": round(float(np.median(t[:, 1] / t[:, 0])), 2),
            "предложений с третями": len(th)}


def main() -> int:
    import librosa
    import soundfile as sf
    import core.pipeline as P
    sim = _mod("sim38", "Инструменты разработки/bench_similarity_38.py")
    sim.KIND = "ecapa"
    name, reader = _sys.argv[1], _sys.argv[2]
    E = [sim.emb(sim.load16(f)) for f in sorted(glob.glob(f"{BANK}/фрагменты/{reader}/*.wav"))]
    ref = np.mean(E, axis=0)
    ref /= np.linalg.norm(ref)
    _os.makedirs(OUT, exist_ok=True)
    rows = []
    if name == "живой":
        d = _os.path.expanduser(f"~/.cache/text2audio/train/{reader}")
        items = [json.loads(l) for l in open(f"{d}/train_clean.jsonl", encoding="utf-8")][::12][:60]
        th, sims, dn, pt = [], [], [], []
        for it in items:
            a, sr = sf.read(it["audio"], dtype="float32")
            a16 = librosa.resample(a, orig_sr=sr, target_sr=16000)
            hw = heard(P, a16)
            _s, per, _m, _n = sentences_with_words(it["text_plain"], hw)
            th += [x for x in (thirds(ws) for ws in per.values()) if x]
            pt += [x for x in (pitch_thirds(a16, ws) for ws in per.values()) if x]
            sims.append(float(sim.emb(a16) @ ref))
        joined = np.concatenate([librosa.resample(sf.read(it["audio"], dtype="float32")[0], orig_sr=24000,
                                                  target_sr=16000) for it in items[:30]])
        r = {"набор": name, "чтец": reader, "вариант": "живой чтец (его фразы)",
             "похожесть": round(float(np.median(sims)), 3), **summarize(th), **pitch_summary(pt),
             **dnsmos(joined)}
        rows.append(r)
    else:
        text = open(_sys.argv[3], encoding="utf-8").read() if _sys.argv[3].endswith(".txt") else None
        files = _sys.argv[4:] if text else _sys.argv[3:]
        for spec in files:
            path, _eq, label = spec.partition("=")
            a16 = sim.load16(path)
            hw = heard(P, a16)
            sents, per, matched, n = sentences_with_words(text, hw)
            th = [x for x in (thirds(ws) for ws in per.values()) if x]
            pt = [x for x in (pitch_thirds(a16, ws) for ws in per.values()) if x]
            sims = []
            for ws in per.values():
                t0, t1 = ws[0][1], ws[-1][2]
                if t1 - t0 >= 1.5:
                    sims.append(float(sim.emb(a16[int(t0 * 16000):int(t1 * 16000)]) @ ref))
            r = {"набор": name, "чтец": reader, "вариант": label or _os.path.basename(path),
                 "файл": path, "с": round(len(a16) / 16000, 1),
                 "похожесть": round(float(np.median(sims)), 3) if sims else None,
                 "слов найдено, %": round(100 * matched / max(n, 1), 1), **summarize(th), **pitch_summary(pt),
                 **dnsmos(a16)}
            rows.append(r)
    with open(f"{OUT}/замеры.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False), flush=True)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
