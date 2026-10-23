"""Ускорение NVIDIA для Windows - отдельной докачкой (как в Scribe 2.4).

В установщике torch собран для процессора: сборка с CUDA весит ~2,5 ГБ, и
класть её всем подряд незачем. Если найдена видеокарта NVIDIA, в Настройках
появляется кнопка: программа скачивает официальную сборку torch с CUDA с
download.pytorch.org (той же версии, что в установщике) и распаковывает её в
папку данных. При следующем запуске activate() ставит эту папку первой в путь
поиска модулей и библиотек - и F5 считает на видеокарте.

На настоящей видеокарте NVIDIA это не проверено: её нет ни у нас, ни на
раннерах GitHub. Проверено только, что ссылка находится (CI).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

TORCH_VERSION = "2.6.0"
CUDA_TAG = "cu124"
INDEX = f"https://download.pytorch.org/whl/{CUDA_TAG}/torch/"


def pack_dir() -> Path:
    from .osutil import model_dir
    return model_dir() / "torch_cuda"


def ready() -> bool:
    return (pack_dir() / "ready.json").is_file()


def activate() -> bool:
    """Подключить докачанный torch с CUDA. Звать ДО первого import torch."""
    d = pack_dir()
    if not ready() or "torch" in sys.modules:
        return False
    sys.path.insert(0, str(d))
    lib = d / "torch" / "lib"
    if lib.is_dir() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    return True


def _open(url: str, timeout: int):
    """HTTPS с сертификатами certifi: в сборке системных может не быть."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=timeout, context=ctx)


def wheel_url(py: str | None = None, plat: str = "win_amd64") -> str:
    """Ссылка на колесо torch с CUDA для этой версии Python."""
    py = py or f"cp{sys.version_info.major}{sys.version_info.minor}"
    with _open(INDEX, 30) as r:
        html = r.read().decode("utf-8", "replace")
    want = f"torch-{TORCH_VERSION}%2B{CUDA_TAG}-{py}-{py}-{plat}.whl"
    for href in re.findall(r'href="([^"]+)"', html):
        if href.split("#")[0].endswith(want):
            href = href.split("#")[0]
            return href if href.startswith("http") else "https://download.pytorch.org" + href
    raise RuntimeError(f"нет сборки torch {TORCH_VERSION}+{CUDA_TAG} для {py} {plat}")


def download(progress=lambda got, total: None, cancel=lambda: False) -> Path:
    """Скачать и распаковать torch с CUDA. Возвращает папку."""
    url = wheel_url()
    dest = pack_dir()
    tmp = dest.with_name("torch_cuda.part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    whl = tmp / "torch.whl"
    with _open(url, 60) as r, open(whl, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            if cancel():
                raise RuntimeError("отменено")
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            progress(got, total)
    with zipfile.ZipFile(whl) as z:
        z.extractall(tmp)
    whl.unlink()
    (tmp / "ready.json").write_text(json.dumps(
        {"torch": TORCH_VERSION, "cuda": CUDA_TAG, "url": url,
         "date": time.strftime("%Y-%m-%d")}), encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return dest


def remove() -> None:
    shutil.rmtree(pack_dir(), ignore_errors=True)


__all__ = ["activate", "download", "ready", "wheel_url", "remove", "pack_dir"]
