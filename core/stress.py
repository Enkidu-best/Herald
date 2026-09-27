"""Расстановка ударений в русском тексте (нейросеть RUAccent).

Ставит ударение автоматически на любой текст - это масштабируемо, ничего не
прописывается вручную. Формат вывода - '+' перед ударной гласной («молок+о»),
именно его понимают Silero и русская модель F5 (accent_tune).

Модуль устойчив к сбоям: если RUAccent недоступен или падает, возвращаем текст
как есть (лучше без ударений, чем уронить всю начитку).
"""

from __future__ import annotations

import re
import threading

_LOCK = threading.Lock()
_ACCENTIZER = None
_TRIED = False


def _get_accentizer():
    """Лениво загрузить RUAccent один раз. При любой ошибке вернуть None."""
    global _ACCENTIZER, _TRIED
    if _ACCENTIZER is not None or _TRIED:
        return _ACCENTIZER
    with _LOCK:
        if _ACCENTIZER is not None or _TRIED:
            return _ACCENTIZER
        _TRIED = True
        try:
            from ruaccent import RUAccent
            acc = RUAccent()
            # по убыванию качества; берём первый, который загрузится
            for kw in (
                dict(omograph_model_size="turbo3.1", use_dictionary=True),
                dict(omograph_model_size="turbo", use_dictionary=True),
                dict(tiny_mode=True, use_dictionary=True),
                dict(),
            ):
                try:
                    acc.load(**kw)
                    _ACCENTIZER = acc
                    break
                except Exception:
                    continue
        except Exception:
            _ACCENTIZER = None
    return _ACCENTIZER


def add_stress(text: str) -> str:
    """Вернуть текст с ударениями ('+' перед ударной гласной). При сбое - как есть."""
    if not text or not text.strip():
        return text
    acc = _get_accentizer()
    if acc is None:
        return text
    try:
        return acc.process_all(text)
    except Exception:
        # RUAccent может спотыкаться на отдельных фрагментах - не роняем начитку
        out = []
        for part in re.split(r"(?<=[.!?…])\s+", text):
            try:
                out.append(acc.process_all(part))
            except Exception:
                out.append(part)
        return " ".join(out)


def to_combining(text_with_plus: str) -> str:
    """'+гласная' -> 'гласная\u0301' (комбинируемое ударение), если движку так удобнее."""
    return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "\\1\u0301", text_with_plus)


def available() -> bool:
    return _get_accentizer() is not None
