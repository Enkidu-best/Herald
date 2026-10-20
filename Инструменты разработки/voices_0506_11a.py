#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть В: разбор голосов 05 (Герасимов) и 06 (Ященко) длинного куска 3.9.

06: похожесть на Ященко (ECAPA к среднему вектору его настоящих фрагментов) по
каждому предложению длинного куска; где голос сменился и совпадает ли смена со
стыком кусков (каждый кусок - отдельный вызов модели с образцом).
Предложения во времени - по меткам слов Whisper, сверенным с текстом главы.

05: манера - темп (слов в минуту, слогов в секунду речи без пауз) и длительность
ударных и безударных гласных (буквы, выровненные CTC-моделью meter2:
jonatasgrosman/wav2vec2-large-xlsr-53-russian) - у живого Герасимова (57
фрагментов его аудиокниги), у 05 и у 01 (голос самой модели) для сравнения.
Под ../.heavy-lock.
    python "Инструменты разработки/voices_0506_11a.py" 06 05
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
LONG = "Сравнение 3.9 - голоса диктора на выбор/Длинный кусок"
SRC = f"{BANK}/длинный кусок/Глава первая (отрывок).txt"
OUT = f"{BANK}/разбор 05 06"
VOW = "аеёиоуыэюя"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sim():
    s = _mod("sim38", "Инструменты разработки/bench_similarity_38.py")
    s.KIND = "ecapa"
    return s


def norm(w):
    return w.lower().replace("ё", "е")


def heard_words(a16: np.ndarray) -> list[tuple[str, float, float]]:
    """Слова Whisper с метками времени - по кускам 30 с (длинный файл целиком не берёт)."""
    import core.pipeline as P
    out = []
    step = 30 * 16000
    for s in range(0, len(a16), step):
        part = a16[s:s + step + 16000]
        for w, t0, t1 in P._listen(part, 16000, "ru", P._ASR_FULL) or []:
            t0, t1 = t0 + s / 16000, t1 + s / 16000
            if out and t0 < out[-1][2] - 0.05:             # перекрытие соседних кусков
                continue
            out.append((w, t0, t1))
    return out


def sentence_times(heard, src_sentences):
    """Предложение текста -> (начало, конец) в записи по совпавшим словам."""
    sw = []
    for si, s in enumerate(src_sentences):
        sw += [(norm(w), si) for w in re.findall(r"[А-Яа-яЁё]+", s)]
    hw = [norm(re.sub(r"[^А-Яа-яЁё]", "", w)) for w, _a, _b in heard]
    sm = difflib.SequenceMatcher(None, [w for w, _ in sw], hw, autojunk=False)
    spans: dict[int, list] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                si = sw[i][1]
                sp = spans.setdefault(si, [heard[j][1], heard[j][2]])
                sp[0], sp[1] = min(sp[0], heard[j][1]), max(sp[1], heard[j][2])
    return spans


def part06() -> dict:
    import core.pipeline as P
    from razdel import sentenize
    sim = _sim()
    frags = sorted(glob.glob(f"{BANK}/фрагменты/Игорь Ященко/*.wav"))
    E = [sim.emb(sim.load16(f)) for f in frags]
    ya = np.mean(E, axis=0)
    ya /= np.linalg.norm(ya)
    d01 = sim.emb(sim.load16(f"{BANK}/образцы/Голос 06.wav"))         # «Голос 01» папки
    a = sim.load16(f"{LONG}/Голос 06.mp3")
    heard = heard_words(a)
    src = open(SRC, encoding="utf-8").read()
    sents = [s.text for s in sentenize(src)]
    spans = sentence_times(heard, sents)
    # куски, как их резал конвейер (600 знаков): номер куска для каждого предложения
    body = P._prepare_text(src, "+", True, True, "ru")
    chunks = P._chunk_paragraphs(body, 600, 0.4, 0.8, 1.1)
    plain = [norm(re.sub(r"[^А-Яа-яЁё ]", "", c[0].replace("+", ""))) for c in chunks]
    rows = []
    for si, s in enumerate(sents):
        if si not in spans:
            continue
        t0, t1 = spans[si]
        if t1 - t0 < 1.5:
            continue
        e = sim.emb(a[int(t0 * 16000):int(t1 * 16000)])
        head = norm(" ".join(re.findall(r"[А-Яа-яЁё]+", s)[:4]))
        ci = next((k for k, p in enumerate(plain) if head and head in p), None)
        rows.append({"№": si + 1, "с": round(t0, 1), "длина": round(t1 - t0, 1), "кусок": ci,
                     "к Ященко": round(float(e @ ya), 2), "к голосу 01": round(float(e @ d01), 2),
                     "начало": s[:60]})
    sims = np.array([r["к Ященко"] for r in rows])
    thr = float(np.median(sims) - 0.12)
    for r in rows:
        r["сменился"] = r["к Ященко"] < thr
    # смена внутри куска или на стыке
    changes = []
    for i in range(1, len(rows)):
        if rows[i]["сменился"] != rows[i - 1]["сменился"]:
            changes.append({"с": rows[i]["с"], "на стыке кусков": rows[i]["кусок"] != rows[i - 1]["кусок"],
                            "куда": "другой диктор" if rows[i]["сменился"] else "снова Ященко"})
    by_chunk = {}
    for r in rows:
        by_chunk.setdefault(r["кусок"], []).append(r["к Ященко"])
    res = {"предложений": len(rows), "медиана к Ященко": round(float(np.median(sims)), 2),
           "порог смены": round(thr, 2), "сменился, предложений": int(sum(r["сменился"] for r in rows)),
           "смен голоса": len(changes),
           "из них на стыке кусков": int(sum(c["на стыке кусков"] for c in changes)),
           "по кускам (среднее к Ященко)": {str(k): round(float(np.mean(v)), 2) for k, v in by_chunk.items()},
           "разброс внутри куска": round(float(np.mean([np.std(v) for v in by_chunk.values() if len(v) > 1])), 3),
           "разброс между кусками": round(float(np.std([np.mean(v) for v in by_chunk.values()])), 3),
           "смены": changes, "предложения": rows}
    print("06:", json.dumps({k: v for k, v in res.items() if k not in ("предложения",)}, ensure_ascii=False),
          flush=True)
    return res


def vowels_of(a16: np.ndarray, text_marked: str):
    """Длительности (с) ударных и безударных гласных по выравниванию букв meter2."""
    from core import meter
    m2 = meter._load()
    lp, _hs = m2.ctc_forward(a16)
    plain = text_marked.replace("+", "")
    al = m2.align(lp, plain)
    words = [w for w in re.findall(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*", text_marked)]
    st, un = [], []
    for w, a in zip(words, al):
        if a is None or a.get("conf", 0) < 0.5:
            continue
        letters = [c for c in w if c != "+"]
        idx = meter.marked_index(w) if "+" in w else None
        vi = -1
        for c, sp in zip(letters, a["letters"]):
            if c.lower() in VOW:
                vi += 1
                if sp is None:
                    continue
                d = (sp[1] - sp[0]) * m2.FRAME
                (st if vi == idx else un).append(d)
    return st, un


def manner(name: str, pieces: list[np.ndarray]) -> dict:
    """Темп и гласные по кускам записи (16 кГц)."""
    import core.pipeline as P
    from core.stress import add_stress_text
    words = sylls = 0
    total = speech = 0.0
    st_all, un_all = [], []
    for a in pieces:
        h = P._listen(a, 16000, "ru", P._ASR_FULL) or []
        if not h:
            continue
        txt = " ".join(w for w, _s, _e in h)
        ws = re.findall(r"[А-Яа-яЁё]+", txt)
        words += len(ws)
        sylls += sum(sum(c in VOW for c in w.lower()) for w in ws)
        total += len(a) / 16000
        fr = 160
        rms = np.sqrt(np.mean(a[: len(a) // fr * fr].reshape(-1, fr) ** 2, axis=1)) + 1e-9
        speech += float(np.sum(20 * np.log10(rms / rms.max()) > -35)) * 0.01
        marked = add_stress_text(" ".join(ws))
        s, u = vowels_of(a, marked)
        st_all += s
        un_all += u
    r = {"минут": round(total / 60, 1), "слов в минуту": round(words / (total / 60), 0),
         "слогов в секунду речи": round(sylls / max(speech, 1e-6), 2),
         "ударная гласная, мс (медиана)": round(1000 * float(np.median(st_all)), 0) if st_all else None,
         "безударная гласная, мс": round(1000 * float(np.median(un_all)), 0) if un_all else None,
         "ударная / безударная": round(float(np.median(st_all) / np.median(un_all)), 2) if st_all and un_all else None,
         "ударных гласных": len(st_all)}
    print(name, json.dumps(r, ensure_ascii=False), flush=True)
    return r


def cut(a16: np.ndarray, sec: float = 12.0) -> list[np.ndarray]:
    """Запись -> куски около sec по паузам (как фрагменты аудиокниги)."""
    fr = 160
    rms = np.sqrt(np.mean(a16[: len(a16) // fr * fr].reshape(-1, fr) ** 2, axis=1)) + 1e-9
    quiet = 20 * np.log10(rms / rms.max()) < -40
    out, start = [], 0
    for i in range(len(quiet)):
        if quiet[i] and (i - start) * 0.01 >= sec:
            out.append(a16[start * fr:i * fr])
            start = i
    return out


def part05() -> dict:
    sim = _sim()
    real = [sim.load16(f) for f in sorted(glob.glob(f"{BANK}/фрагменты/Вячеслав Герасимов/*.wav"))]
    return {"Герасимов (аудиокнига)": manner("Герасимов", real),
            "05 (клон)": manner("05", cut(sim.load16(f"{LONG}/Голос 05.mp3"))),
            "01 (голос модели)": manner("01", cut(sim.load16(f"{LONG}/Голос 01.mp3")))}


def main() -> int:
    _os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/итог.json"
    res = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    for p in _sys.argv[1:] or ["06", "05"]:
        res[p] = part06() if p == "06" else part05()
        json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
