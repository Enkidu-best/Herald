# -*- coding: utf-8 -*-
"""Корзина программы (задание 12, «Единый стандарт» 3.2): удалённое лежит 30 дней,
его можно вернуть на место; «Удалить навсегда» - только отсюда.

Что попадает: файлы удалённого голоса (образец, расшифровка, поправки, ссылка на свои
веса - сами веса обучения не трогаются) и служебные файлы готовой книги («_служебное»).
Каждая запись - папка `~/.cache/text2audio/trash/<время>__<имя>/` с файлами и
`item.json`: что это, откуда (пути на место), когда удалено, размер.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass

TRASH_DIR = os.path.expanduser("~/.cache/text2audio/trash")
KEEP_DAYS = 30
META = "item.json"


@dataclass
class Item:
    id: str              # имя папки в корзине
    kind: str            # "voice" | "service"
    title: str           # что показать: имя голоса, название книги
    deleted: float       # когда удалено (time.time())
    size: int            # байт
    origins: dict        # имя файла в корзине -> путь, куда вернуть


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    n = 0
    for root, _d, files in os.walk(path):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def put(paths: list[str], kind: str, title: str) -> Item | None:
    """Переложить файлы / папки в корзину одной записью. Нет ни одного - None."""
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    safe = "".join(c if c.isalnum() or c in " -_." else "_" for c in title)[:60].strip() or kind
    iid = f"{time.strftime('%Y%m%d-%H%M%S')}__{safe}__{uuid.uuid4().hex[:6]}"
    d = os.path.join(TRASH_DIR, iid)
    os.makedirs(d, exist_ok=True)
    origins, size = {}, 0
    for i, p in enumerate(paths):
        name = f"{i:02d}__{os.path.basename(p.rstrip(os.sep))}"
        size += _size(p)
        shutil.move(p, os.path.join(d, name))
        origins[name] = os.path.abspath(p)
    item = Item(iid, kind, title, time.time(), size, origins)
    with open(os.path.join(d, META), "w", encoding="utf-8") as f:
        json.dump(item.__dict__, f, ensure_ascii=False, indent=1)
    return item


def items() -> list[Item]:
    """Всё, что в корзине, - новые сверху."""
    out = []
    if not os.path.isdir(TRASH_DIR):
        return out
    for iid in os.listdir(TRASH_DIR):
        p = os.path.join(TRASH_DIR, iid, META)
        try:
            with open(p, encoding="utf-8") as f:
                out.append(Item(**json.load(f)))
        except (OSError, ValueError, TypeError):
            continue                    # не наша папка или испорченная запись - не показываем
    return sorted(out, key=lambda x: -x.deleted)


def conflicts(item: Item) -> list[str]:
    """Пути, где на месте уже что-то есть (вернуть - значит заменить)."""
    return [o for o in item.origins.values() if os.path.exists(o)]


def restore(item: Item, replace: bool = False) -> None:
    """Вернуть на прежнее место. Папки, куда возвращать, нет - FileNotFoundError."""
    d = os.path.join(TRASH_DIR, item.id)
    for name, origin in item.origins.items():
        if not os.path.isdir(os.path.dirname(origin)):
            raise FileNotFoundError(os.path.dirname(origin))
    for name, origin in item.origins.items():
        src = os.path.join(d, name)
        if not os.path.exists(src):
            continue
        if os.path.exists(origin):
            if not replace:
                raise FileExistsError(origin)
            if os.path.isdir(origin):
                shutil.rmtree(origin)
            else:
                os.remove(origin)
        shutil.move(src, origin)
    shutil.rmtree(d, ignore_errors=True)


def purge(item: Item) -> None:
    """Удалить навсегда."""
    shutil.rmtree(os.path.join(TRASH_DIR, item.id), ignore_errors=True)


def purge_old(days: int = KEEP_DAYS) -> int:
    """Старше срока - удалить навсегда (зовётся при запуске программы)."""
    n, edge = 0, time.time() - days * 86400
    for it in items():
        if it.deleted < edge:
            purge(it)
            n += 1
    return n
