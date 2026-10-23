"""Кэш синтеза: готовый звук куска по хешу «текст + голос + настройки движка».

Зачем. Кусок книги считается десятки секунд, а пересчитывать его приходится
часто без всякой нужды: поменяли паузы или формат - и часть озвучивается
заново; «Проверить голос» дважды на той же книге; повторы в тексте. Звук куска
от пауз и формата не зависит, поэтому он берётся из кэша, а заново собирается
только сама часть - за секунды вместо часов.

Ключ - sha1 от движка (модель, шаги, обработка хвоста - engine.cache_key()),
файлов образца голоса, текста куска (уже с ударениями) и его темпа. В кэш
кладётся только кусок, прошедший проверку краёв. Лежит в
~/.cache/text2audio/synth/ во FLAC; старше всего удаляется сверх MAX_BYTES.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from .log import log_problem

from .osutil import data_dir as _data_dir
CACHE_DIR = os.path.join(str(_data_dir()), "synth")
MAX_BYTES = 3 * 1024 ** 3          # ~3 ГБ - это около 30 часов звука
_voice_hash: dict[tuple, str] = {}


def _voice_files_hash(voices_dir: str, voice: str) -> str:
    """Хеш файлов образца: перезаписали образец - старые куски не годятся."""
    parts = []
    for ext in (".wav", ".txt"):
        p = os.path.join(voices_dir, voice + ext)
        try:
            st = os.stat(p)
        except OSError:
            continue
        k = (p, st.st_mtime_ns, st.st_size)
        if k not in _voice_hash:
            with open(p, "rb") as f:
                _voice_hash[k] = hashlib.sha1(f.read()).hexdigest()
        parts.append(_voice_hash[k])
    # свои веса голоса F5 (11б): у «… (F5, обучен)» образец тот же, что у обычного голоса -
    # без этого кэш отдал бы звук прежней модели; переобучили - старые куски не годятся
    p = os.path.join(voices_dir, voice + ".f5model")
    if os.path.isfile(p):
        w = os.path.join(os.path.expanduser(open(p, encoding="utf-8").read().strip()), "model_v1.safetensors")
        try:
            st = os.stat(w)
            parts.append(f"f5model:{w}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append("f5model:нет")
    return ",".join(parts)


def key(engine_key: str, voices_dir: str, voice: str, text: str, speed_mul: float) -> str:
    raw = "\n".join((engine_key, voice, _voice_files_hash(voices_dir, voice),
                     f"{speed_mul:.4f}", text))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _path(k: str) -> str:
    return os.path.join(CACHE_DIR, k[:2], k + ".flac")


def get(k: str) -> tuple[np.ndarray, list | None] | None:
    """(звук, распознанные слова с метками времени или None) - или None."""
    p = _path(k)
    if not os.path.exists(p):
        return None
    try:
        import soundfile as sf
        a, _sr = sf.read(p, dtype="float32")
        os.utime(p)                       # свежий - значит, чистка удалит его последним
        words = None
        if os.path.exists(p[:-5] + ".json"):
            with open(p[:-5] + ".json", encoding="utf-8") as f:
                words = [tuple(w) for w in json.load(f)]
        return a.reshape(-1), words
    except Exception as e:
        log_problem("кусок из кэша синтеза не прочитался, считаю заново", e, show=False)
        return None


def put(k: str, audio: np.ndarray, sr: int, words: list | None = None) -> None:
    """Положить кусок; words - распознанные слова (нужны паузам, см. pipeline)."""
    if not len(audio):
        return
    p = _path(k)
    try:
        import soundfile as sf
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if words is not None:
            with open(p[:-5] + ".json", "w", encoding="utf-8") as f:
                json.dump([list(w) for w in words], f, ensure_ascii=False)
        tmp = p + ".part.flac"
        sf.write(tmp, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_24")
        os.replace(tmp, p)
    except Exception as e:
        log_problem("кусок не записался в кэш синтеза", e, show=False)


def prune(max_bytes: int = MAX_BYTES) -> int:
    """Удалить самые старые куски сверх лимита. Возвращает, сколько удалено."""
    files, files_side = [], []
    for root, _d, names in os.walk(CACHE_DIR):
        for n in names:
            p = os.path.join(root, n)
            try:
                st = os.stat(p)
            except OSError:
                continue
            if n.endswith(".flac"):
                files.append((st.st_mtime, st.st_size, p))
            else:
                total_side = st.st_size
                files_side.append(total_side)
    total = sum(f[1] for f in files) + sum(files_side)
    removed = 0
    for _m, size, p in sorted(files):
        if total <= max_bytes:
            break
        try:
            os.remove(p)
            total -= size
            removed += 1
            side = p[:-5] + ".json"            # слова куска - вместе с ним
            if p.endswith(".flac") and os.path.exists(side):
                total -= os.path.getsize(side)
                os.remove(side)
        except OSError:
            pass                          # занят или уже удалён - не важно
    return removed
