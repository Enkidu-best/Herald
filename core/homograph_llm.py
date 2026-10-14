"""Третий судья для омографов: локальная языковая модель через Ollama.

Слово-омограф и предложение целиком -> модель выбирает один из вариантов
ударения, которые знает расстановщик (за́мок / замо́к). Словарей нет: модель
понимает смысл фразы. Всё локально, текст никуда не уходит.

Включается, только если доказал пользу на замере (см. docs/Приёмка 3.0), и
только там, где RUAccent и silero-stress разошлись: так на книгу уходят минуты,
а не часы. Перед синтезом модель выгружается из памяти (keep_alive=0) - иначе
она отнимает у синтеза 6-7 ГБ.
"""

from __future__ import annotations

import json
import urllib.request

OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:9b"
FALLBACK_MODEL = "qwen3.5:4b"
TIMEOUT = 60

_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def _post(path: str, data: dict, timeout: float = TIMEOUT) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(data).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def available_model() -> str | None:
    """Какая из моделей установлена (9b, иначе 4b), или None."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as r:
            names = {m["name"] for m in json.loads(r.read())["models"]}
    except Exception:
        return None
    for m in (MODEL, FALLBACK_MODEL):
        if m in names:
            return m
    return None


def upper_stress(variant: str) -> str:
    """«зам+ок» -> «замОк»: так модели понятнее, чем значок «+»."""
    out, up = [], False
    for c in variant:
        if c == "+":
            up = True
            continue
        out.append(c.upper() if up and c in _VOWELS else c)
        if c in _VOWELS:
            up = False
    return "".join(out)


def choose(sentence: str, variants: list[str], model: str | None = None) -> int | None:
    """Номер верного варианта (0, 1, …) по смыслу предложения или None."""
    model = model or available_model()
    if not model or len(variants) < 2:
        return None
    word = variants[0].replace("+", "")
    opts = "\n".join(f"{i + 1}) {upper_stress(v)}" for i, v in enumerate(variants))
    prompt = (f"Предложение: «{sentence}»\n"
              f"Слово «{word}» в этом предложении - как оно произносится?\n"
              f"Варианты (ударная гласная - заглавная):\n{opts}\n"
              "Выбери вариант, верный по смыслу и грамматике именно этого предложения. "
              "Ответь только номером.")
    try:
        r = _post("/api/chat", {
            "model": model, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 8},
            "format": {"type": "object", "properties": {"answer": {"type": "integer"}},
                       "required": ["answer"]},
            "messages": [
                {"role": "system", "content": "Ты корректор русских аудиокниг и точно знаешь "
                                              "нормы ударения."},
                {"role": "user", "content": prompt}]})
        n = int(json.loads(r["message"]["content"])["answer"]) - 1
        return n if 0 <= n < len(variants) else None
    except Exception:
        return None


def unload(model: str | None = None) -> None:
    """Выгрузить модель из памяти перед синтезом."""
    for m in ([model] if model else [MODEL, FALLBACK_MODEL]):
        try:
            _post("/api/generate", {"model": m, "keep_alive": 0}, timeout=10)
        except Exception:
            pass                # не была загружена - это и нужно
