#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11, пп. 2 и 4: длинный кусок (7-8 мин) полным конвейером Herald.

Отрывок «Глава первая» из «Жажда власти» (Книги/) - через обычный convert_book:
паузы после предложения / абзаца / между репликами (как в Настройках), проверка
слов по звуку и meter2 с пересчётом, ударения RUAccent, один голосовой промпт и одно
зерно на всю главу. Голоса - из папки «Сравнение 3.9 - голоса диктора на выбор»:
01, 03, 02 («голоса диктора» самой ru-stress) и 05, 06 (Герасимов и Ященко с
усреднённым вектором голоса). Готовый файл - в «…/Длинный кусок/Голос NN.mp3».
Под ../.heavy-lock.
    python "Инструменты разработки/long_piece_39.py" [01 03 02 05 06]
    python "Инструменты разработки/long_piece_39.py" f5      # 11а, Г: Ященко и Герасимов движком F5
    python "Инструменты разработки/long_piece_39.py" обучен  # 11а, Д: обученные голоса Qwen
    python "Инструменты разработки/long_piece_39.py" f5обучен  # 11б, шаг 4: дообученный F5
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(_sys.path[0])

import glob
import json
import shutil
import time

BANK = "Архив/голоса диктора 3.9"
DST = "Сравнение 3.9 - голоса диктора на выбор/Длинный кусок"
BOOK = "Книги/Жажда власти 1 - Сергей Тармашев/Жажда власти. Книга 1.fb2"
# номер в папке Андрея -> (голос для движка, папка фрагментов для среднего вектора)
VOICES = {
    "01": (_os.path.abspath(f"{BANK}/образцы/Голос 06.wav"), None),
    "02": (_os.path.abspath(f"{BANK}/образцы/Голос 05.wav"), None),
    "03": (_os.path.abspath(f"{BANK}/образцы/Голос 03.wav"), None),
    # Герасимову - «живой» образец (размах тона 9,1 -> 14,2 полутона на абзаце, п. 3)
    "05": (_os.path.abspath(f"{BANK}/интонация/живой образец - Вячеслав Герасимов.wav"),
           _os.path.abspath(f"{BANK}/фрагменты/Вячеслав Герасимов")),
    "06": ("Игорь Ященко", _os.path.abspath(f"{BANK}/фрагменты/Игорь Ященко")),
}


def excerpt() -> str:
    """«Глава первая» до ~6500 знаков, по границе абзаца."""
    from core.readers import read_document
    from core.chapters import build_chapters
    out = _os.path.join(BANK, "длинный кусок", "Глава первая (отрывок).txt")
    if _os.path.exists(out):
        return out
    ch = [c for c in build_chapters(read_document(BOOK), 15) if c.title == "Глава первая"][0]
    paras, n = [], 0
    for p in ch.text.split("\n"):
        if not p.strip():
            continue
        if n > 6500:
            break
        paras.append(p.strip())
        n += len(p)
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    body = "\n\n".join(paras)
    if not body.startswith("Глава первая"):
        body = "Глава первая\n\n" + body
    open(out, "w", encoding="utf-8").write(body + "\n")
    return out


# 11а, часть Г: «старая сборка» - движок F5 (профиль «Основной»), голоса чтецов
F5_VOICES = {"Игорь Ященко": "Ященко (F5)", "Вячеслав Герасимов": "Герасимов (F5)"}


def main_f5() -> int:
    import core.pipeline as P
    from core.profiles import PROFILES, engine_kwargs
    src = excerpt()
    dst = _os.path.join(DST, "Старая сборка F5")
    _os.makedirs(dst, exist_ok=True)
    p = PROFILES["main"]
    stats = {}
    for voice, name in F5_VOICES.items():
        work = _os.path.abspath(_os.path.join(BANK, "длинный кусок", name))
        shutil.rmtree(work, ignore_errors=True)
        opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60,
                                make_m4b=False, make_chapter_mp3=True, sound_check=True, narrator=voice)
        t0 = time.time()
        res = P.convert_book(src, engine=p.engine, eng_kwargs=engine_kwargs(p), voice=voice,
                             out_dir=work, options=opts)
        took = time.time() - t0
        mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
        shutil.copy(mp3, _os.path.join(dst, f"{name}.mp3"))
        stats[name] = {"звук, с": round(res.duration, 1), "счёт, с": round(took, 1),
                       "на секунду звука": round(took / max(res.duration, 1), 2)}
        print(f"{name}: {json.dumps(stats[name], ensure_ascii=False)}", flush=True)
        P.unload_engines()
    path = _os.path.join(BANK, "длинный кусок", "замер.json")
    old = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    old.update(stats)
    json.dump(old, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


# 11б, шаг 4: F5, дообученный на записях чтеца (свои веса голоса - «<голос>.f5model»)
F5_TRAINED = {"Игорь Ященко (F5, обучен)": "Ященко (F5, обучен)"}


def main_f5_trained() -> int:
    import core.pipeline as P
    from core.profiles import PROFILES, engine_kwargs
    src = excerpt()
    dst = _os.path.join(DST, "Дообученный F5")
    _os.makedirs(dst, exist_ok=True)
    p = PROFILES["main"]
    stats = {}
    for voice, name in F5_TRAINED.items():
        work = _os.path.abspath(_os.path.join(BANK, "длинный кусок", name))
        shutil.rmtree(work, ignore_errors=True)
        opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60,
                                make_m4b=False, make_chapter_mp3=True, sound_check=True, narrator=voice)
        t0 = time.time()
        res = P.convert_book(src, engine=p.engine, eng_kwargs=engine_kwargs(p), voice=voice,
                             out_dir=work, options=opts)
        took = time.time() - t0
        mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
        shutil.copy(mp3, _os.path.join(dst, f"{name}.mp3"))
        stats[name] = {"звук, с": round(res.duration, 1), "счёт, с": round(took, 1),
                       "на секунду звука": round(took / max(res.duration, 1), 2)}
        print(f"{name}: {json.dumps(stats[name], ensure_ascii=False)}", flush=True)
        P.unload_engines()
    return 0


# 11а, часть Д: обученные голоса (ru-stress + адаптер чтеца, install_trained_11a.py)
TRAINED = {"Игорь Ященко (обучен)": "Ященко (обучен)", "Вячеслав Герасимов (обучен)": "Герасимов (обучен)"}


def main_trained() -> int:
    import core.pipeline as P
    from core.profiles import PROFILES, engine_kwargs
    src = excerpt()
    dst = _os.path.join(DST, "Обученный голос")
    _os.makedirs(dst, exist_ok=True)
    p = PROFILES["qwen"]
    stats = {}
    for voice, name in TRAINED.items():
        work = _os.path.abspath(_os.path.join(BANK, "длинный кусок", name))
        shutil.rmtree(work, ignore_errors=True)
        opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60,
                                make_m4b=False, make_chapter_mp3=True, sound_check=True, narrator=voice)
        t0 = time.time()
        res = P.convert_book(src, engine=p.engine, eng_kwargs=engine_kwargs(p), voice=voice,
                             out_dir=work, options=opts)
        took = time.time() - t0
        mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
        shutil.copy(mp3, _os.path.join(dst, f"{name}.mp3"))
        stats[name] = {"звук, с": round(res.duration, 1), "счёт, с": round(took, 1),
                       "на секунду звука": round(took / max(res.duration, 1), 2)}
        print(f"{name}: {json.dumps(stats[name], ensure_ascii=False)}", flush=True)
        P.unload_engines()
    path = _os.path.join(BANK, "длинный кусок", "замер.json")
    old = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    old.update(stats)
    json.dump(old, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


def main() -> int:
    import core.pipeline as P
    if _sys.argv[1:] == ["f5"]:
        return main_f5()
    if _sys.argv[1:] == ["обучен"]:
        return main_trained()
    if _sys.argv[1:] == ["f5обучен"]:
        return main_f5_trained()
    src = excerpt()
    _os.makedirs(DST, exist_ok=True)
    stats = {}
    for num in _sys.argv[1:] or list(VOICES):
        voice, centroid = VOICES[num]
        work = _os.path.abspath(_os.path.join(BANK, "длинный кусок", f"Голос {num}"))  # как в окне - путь целиком
        shutil.rmtree(work, ignore_errors=True)
        # п. 3: кусок до 600 знаков и temperature 1,0 / top_p 0,95 - живее и без потерь слов
        kw = {"model": "qwen-stress", "bits": 8, "seed": 7, "max_chars": 600,
              "temperature": 1.0, "top_p": 0.95}
        if centroid:
            kw["centroid"] = centroid
        opts = P.ConvertOptions(use_cache=False, single_file=True, minutes_per_file=60,
                                make_m4b=False, make_chapter_mp3=True, sound_check=True,
                                narrator=f"Голос {num}")
        t0 = time.time()
        res = P.convert_book(src, engine="mlxaudio", eng_kwargs=kw, voice=voice, out_dir=work, options=opts)
        took = time.time() - t0
        mp3 = sorted(glob.glob(_os.path.join(work, "**", "*.mp3"), recursive=True))[0]
        shutil.copy(mp3, _os.path.join(DST, f"Голос {num}.mp3"))
        stats[num] = {"звук, с": round(res.duration, 1), "счёт, с": round(took, 1),
                      "на секунду звука": round(took / max(res.duration, 1), 2)}
        print(f"Голос {num}: {json.dumps(stats[num], ensure_ascii=False)}", flush=True)
        P.unload_engines()
    path = _os.path.join(BANK, "длинный кусок", "замер.json")
    old = json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}
    old.update(stats)
    json.dump(old, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    rc = main()
    _sys.stdout.flush()
    _os._exit(rc)
