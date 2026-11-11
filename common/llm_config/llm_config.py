"""Одна настройка модели Ollama для всех программ (стандарт, раздел 7; задание «_Общая база 01», 11.11.2026).

Порядок (первое непустое значение):
  1) переменная окружения программы (IC_LLM_MODEL, CLEARDOC_LLM_MODEL, TRIAGE_MODEL …) - `env=`;
  2) выбор пользователя в Настройках программы - `user=`;
  3) файл `.models.json` - ищется вверх от папки копии модуля (`<проект>/common/llm_config` → `Разработка ПО/`);
  4) встроенное значение (`DEFAULTS`) - программа работает и без файла (GitHub, CI, сборка Windows).

Файл: {"main": "qwen3.5:9b", "fallback": "qwen3.5:4b", "embed": "bge-m3", "ollama_url": "http://127.0.0.1:11434"}.
Файл читается при каждом вызове (он маленький) - новое значение действует с ближайшего вызова или перезапуска программы.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

FILE_NAME = ".models.json"
FILE_ENV = "LLM_MODELS_FILE"          # явный путь к файлу (тесты, особые запуски); пустая строка - файл не читать
DEFAULTS = {"main": "qwen3.5:9b", "fallback": "qwen3.5:4b", "embed": "bge-m3"}
DEFAULT_URL = "http://127.0.0.1:11434"
URL_KEY = "ollama_url"

_HERE = Path(__file__).resolve().parent
log = logging.getLogger(__name__)
_warned: set[str] = set()


def find_file(start: str | Path | None = None) -> Path | None:
    """Путь к `.models.json`: из `LLM_MODELS_FILE` или первый найденный вверх от `start` (по умолчанию - папка модуля)."""
    if FILE_ENV in os.environ:
        p = os.environ[FILE_ENV].strip()
        return Path(p).expanduser() if p else None
    d = Path(start).expanduser().resolve() if start else _HERE
    if d.is_file():
        d = d.parent
    for x in (d, *d.parents):
        f = x / FILE_NAME
        if f.is_file():
            return f
    return None


def _warn(key: str, text: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(text)


def load(start: str | Path | None = None) -> dict[str, str]:
    """Значения файла (только непустые строки). Файла нет или он битый - пустой словарь и одно предупреждение в журнал."""
    f = find_file(start)
    if f is None:
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        _warn(f"{f}:{e}", f"{f}: не читается ({e.__class__.__name__}: {e}) - беру встроенные значения")
        return {}
    if not isinstance(data, dict):
        _warn(f"{f}:type", f"{f}: ожидался объект JSON {{\"main\": …}} - беру встроенные значения")
        return {}
    return {k: v.strip() for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and v.strip()}


def _from_env(env) -> tuple[str, str] | None:
    for name in ([env] if isinstance(env, str) else list(env or ())):
        v = os.environ.get(name, "").strip()
        if v:
            return v, f"env {name}"
    return None


def resolve(role: str = "main", *, env=None, user: str | None = None, default: str | None = None,
            start: str | Path | None = None) -> tuple[str, str]:
    """(значение, откуда): «env ИМЯ», «user», «file <путь>», «default»."""
    hit = _from_env(env)
    if hit:
        return hit
    if isinstance(user, str) and user.strip():
        return user.strip(), "user"
    v = load(start).get(role)
    if v:
        return v, f"file {find_file(start)}"
    d = default or (DEFAULT_URL if role == URL_KEY else DEFAULTS.get(role))
    if not d:
        raise KeyError(f"llm_config: нет роли «{role}» ни в файле, ни во встроенных значениях")
    return d, "default"


def model(role: str = "main", *, env=None, user: str | None = None, default: str | None = None,
          start: str | Path | None = None) -> str:
    """Модель для роли: main - рабочая, fallback - запасная, embed - эмбеддинги."""
    return resolve(role, env=env, user=user, default=default, start=start)[0]


def ollama_url(*, env=None, default: str | None = None, start: str | Path | None = None) -> str:
    """Адрес Ollama без «/» в конце."""
    return resolve(URL_KEY, env=env, default=default, start=start)[0].rstrip("/")


def main(argv: list[str] | None = None) -> int:
    """`python -m common.llm_config` - какие значения видит программа и откуда."""
    rows = {r: resolve(r) for r in (*DEFAULTS, URL_KEY)}
    print(json.dumps({"file": str(find_file() or ""), **{r: {"value": v, "from": s} for r, (v, s) in rows.items()}},
                     ensure_ascii=False, indent=1))
    return 0
