#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11а, часть Д: обученный голос против клона по образцу (05 / 06) и F5.

Длинный кусок («Глава первая», полный конвейер) каждого варианта:
  * похожесть на чтеца (ECAPA к среднему вектору его настоящих фрагментов) по
    каждому предложению - как в части В (voices_0506_11a.py): медиана, доля
    предложений ниже 0,50 (порог смены у 06), смены голоса (порог - медиана - 0,12);
  * слова: доля слов главы, которые Whisper не услышал (грубо: числа Whisper пишет
    цифрами - они считаются неуслышанными у всех одинаково); окончания - доля
    последних слов предложений, услышанных целиком;
  * темп - слогов услышанных слов в секунду речи (без пауз); у живых чтецов на их записях
    5,8 / 5,6 (там текст совпадает со звуком);
  * манера Герасимова: отношение ударной гласной к безударной (у живого ~2,0).
Пишет «Архив/голоса диктора 3.9/обученный голос/итог.json». Под ../.heavy-lock.
    python "Инструменты разработки/lora_eval_11a.py"
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
OUT = f"{BANK}/обученный голос"
RUNS = {
    "Игорь Ященко": {"06 (клон по образцу)": f"{LONG}/Голос 06.mp3",
                     "F5": f"{LONG}/Старая сборка F5/Ященко (F5).mp3",
                     "обученный": f"{LONG}/Обученный голос/Ященко (обучен).mp3",
                     "обученный, только вектор": f"{LONG}/Обученный голос/Ященко (обучен, только вектор).mp3"},
    "Вячеслав Герасимов": {"05 (клон по образцу)": f"{LONG}/Голос 05.mp3",
                           "F5": f"{LONG}/Старая сборка F5/Герасимов (F5).mp3",
                           "обученный": f"{LONG}/Обученный голос/Герасимов (обучен).mp3",
                           "обученный, только вектор": f"{LONG}/Обученный голос/Герасимов (обучен, только вектор).mp3"},
}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


V = _mod("v0506", "Инструменты разработки/voices_0506_11a.py")


def speech_sec(a16) -> float:
    fr = 160
    rms = np.sqrt(np.mean(a16[: len(a16) // fr * fr].reshape(-1, fr) ** 2, axis=1)) + 1e-9
    return float(np.sum(20 * np.log10(rms / rms.max()) > -35)) * 0.01


def similarity(path: str, ref: np.ndarray, sim, sents) -> dict:
    a = sim.load16(path)
    heard = V.heard_words(a)
    spans = V.sentence_times(heard, sents)
    vals = []
    for si in sorted(spans):
        t0, t1 = spans[si]
        if t1 - t0 >= 1.5:
            vals.append(float(sim.emb(a[int(t0 * 16000):int(t1 * 16000)]) @ ref))
    v = np.array(vals)
    thr = float(np.median(v) - 0.12)
    low = v < thr
    src_w, last_i = [], []
    for s in sents:
        ws = [V.norm(w) for w in re.findall(r"[А-Яа-яЁё]+", s)]
        src_w += ws
        if ws:
            last_i.append(len(src_w) - 1)                     # последнее слово предложения
    hw = [V.norm(re.sub(r"[^А-Яа-яЁё]", "", w)) for w, _a, _b in heard]
    sm = difflib.SequenceMatcher(None, src_w, hw, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    hit = set()
    for b in sm.get_matching_blocks():
        hit.update(range(b.a, b.a + b.size))
    # окончание слова: последнее слово предложения услышано целиком (съеденное окончание -
    # другое слово для сверки: «правил» против «правило»)
    ends_ok = sum(i in hit for i in last_i)
    # темп - по УСЛЫШАННЫМ словам: по тексту пропуски слов выглядели бы как быстрая речь
    syl = sum(sum(c in "аеёиоуыэюя" for c in w) for w in hw)
    return {"минут": round(len(a) / 16000 / 60, 1), "предложений": len(v),
            "медиана похожести": round(float(np.median(v)), 2),
            "10-й процентиль": round(float(np.percentile(v, 10)), 2),
            "ниже 0,50": int(np.sum(v < 0.50)),
            "смен голоса (медиана - 0,12)": int(np.sum(low[1:] != low[:-1])),
            "слов не услышано, %": round(100 * (1 - matched / max(len(src_w), 1)), 1),
            "последних слов предложений услышано, %": round(100 * ends_ok / max(len(last_i), 1), 1),
            "слогов в секунду речи": round(syl / max(speech_sec(a), 1e-6), 2)}, a


def main() -> int:
    from razdel import sentenize
    sim = V._sim()
    sents = [s.text for s in sentenize(open(V.SRC, encoding="utf-8").read())]
    _os.makedirs(OUT, exist_ok=True)
    path0 = f"{OUT}/итог.json"
    res = json.load(open(path0, encoding="utf-8")) if _os.path.exists(path0) else {}
    for reader, runs in RUNS.items():
        E = [sim.emb(sim.load16(f)) for f in sorted(glob.glob(f"{BANK}/фрагменты/{reader}/*.wav"))]
        ref = np.mean(E, axis=0)
        ref /= np.linalg.norm(ref)
        res.setdefault(reader, {})
        for name, path in runs.items():
            if _sys.argv[1:] and name not in _sys.argv[1:]:          # только эти варианты
                continue
            if not _os.path.exists(path):
                print(reader, name, "нет файла", path, flush=True)
                continue
            r, a = similarity(path, ref, sim, sents)
            if reader == "Вячеслав Герасимов":
                m = V.manner(name, V.cut(a))
                r["ударная / безударная"] = m["ударная / безударная"]
            res[reader][name] = r
            print(reader, name, json.dumps(r, ensure_ascii=False), flush=True)
            json.dump(res, open(f"{OUT}/итог.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    old = json.load(open(f"{BANK}/разбор 05 06/итог.json", encoding="utf-8")).get("05", {})
    res["Герасимов (аудиокнига), из части В"] = old.get("Герасимов (аудиокнига)")
    json.dump(res, open(f"{OUT}/итог.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
