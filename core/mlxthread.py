# -*- coding: utf-8 -*-
"""Одна нить MLX на весь процесс Herald (задание 14 В; находка Scribe 06, 23.10).

MLX привязывает загруженную модель, скомпилированные графы и потоки вычислений («Stream(gpu, …)») к нити, в которой
они созданы. Вызов из другой нити падает с «There is no Stream(gpu, 1) in current thread», а после сбоя в кэше
остаются массивы прежней нити. Поэтому всё MLX в процессе - F5-MLX (загрузка, образец, синтез), проверка слов по
звуку (mlx_whisper), очистка кэша при выгрузке - идёт через одну постоянную нить: run(fn, …). Qwen (mlx-audio) - в
отдельном процессе и сюда не относится.

  * вызов изнутри нити MLX выполняется сразу (иначе одна нить ждала бы сама себя);
  * ошибка - кэш MLX очищается в той же нити, ошибка уходит вызывающему.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

NAME = "mlx"
_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix=NAME)


def in_thread() -> bool:
    return threading.current_thread().name.startswith(NAME + "_")


def _clear() -> None:
    try:
        import mlx.core as mx
        getattr(mx, "clear_cache", getattr(getattr(mx, "metal", None), "clear_cache", lambda: None))()
    except Exception:
        pass                          # не Mac или старый MLX - нечего чистить


def clear_cache() -> None:
    """Очистить кэш MLX - в нити MLX."""
    if in_thread():
        _clear()
    else:
        _POOL.submit(_clear).result()


def run(fn, *args, **kw):
    """Выполнить fn в нити MLX и вернуть результат; ошибка - очистка кэша и та же ошибка."""
    if in_thread():
        return fn(*args, **kw)

    def job():
        try:
            return fn(*args, **kw)
        except BaseException:
            _clear()
            raise
    return _POOL.submit(job).result()


def thread_ident() -> int:
    """Номер нити MLX (для проверок: все вызовы - в одной нити)."""
    return _POOL.submit(threading.get_ident).result()


__all__ = ["run", "clear_cache", "in_thread", "thread_ident"]
