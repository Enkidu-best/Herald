#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Проверка репозитория Herald перед отправкой на GitHub - по ВСЕЙ истории ветки
(все версии всех файлов), по образцу check_repo.py других проектов Андрея.

    python "Инструменты разработки/check_repo.py"          # ветка main
    python "Инструменты разработки/check_repo.py" HEAD     # текущая

Ищет:
- ключи и токены (sk-, hf_, ghp_, AKIA, «token=», «api_key=»), файлы .env;
- пути Mac пользователя (/Users/<имя>/), почтовые адреса;
- звук, книги и образцы голосов (*.wav *.mp3 *.flac *.m4a *.m4b *.ogg *.fb2 *.epub,
  папки «Образцы голосов», «Книги», «Живая проверка»);
- веса моделей (*.safetensors *.onnx *.pt *.pth *.bin *.npz *.gguf).
Печатает только числа и имена файлов (не содержимое). Код 0 - чисто, 1 - найдено.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_EXT = (".wav", ".mp3", ".flac", ".m4a", ".m4b", ".ogg", ".fb2", ".epub",
           ".safetensors", ".onnx", ".pt", ".pth", ".bin", ".npz", ".gguf")
BAD_DIRS = ("Образцы голосов/", "Книги/", "Живая проверка/", ".env")
SECRET = re.compile(rb"(sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|"
                    rb"AKIA[0-9A-Z]{16}|(?i:api[_-]?key|secret|token)\s*[=:]\s*['\"][^'\"\s]{12,})")
USERS = re.compile(rb"/Users/[A-Za-z][\w.-]+/")
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@(?!example\.|users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[a-z]{2,}")
ALLOWED_EMAIL = (b"noreply@anthropic.com",)
# часть программы (детектор речи Silero VAD, MIT) - в репозитории намеренно
ALLOWED_FILES = {"assets/models/silero_vad.onnx"}
BINARY = (".png", ".jpg", ".jpeg", ".ico", ".icns", ".gif", ".pdf", ".docx", ".xlsx", ".zip")


def git(*a: str) -> bytes:
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, check=True).stdout


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else "main"
    blobs: dict[bytes, str] = {}
    for line in git("rev-list", "--objects", ref).splitlines():
        parts = line.split(b" ", 1)
        if len(parts) == 2:
            blobs.setdefault(parts[0], parts[1].decode("utf-8", "replace"))
    files_bad, secrets, users, emails = set(), set(), set(), set()
    for sha, path in blobs.items():
        low = path.lower()
        if path in ALLOWED_FILES:
            continue
        if low.endswith(BAD_EXT) or any(d.lower() in low for d in BAD_DIRS):
            files_bad.add(path)
            continue
        try:
            kind = git("cat-file", "-t", sha.decode()).strip()
        except subprocess.CalledProcessError:
            continue
        if kind != b"blob" or low.endswith(BINARY):
            continue
        data = git("cat-file", "-p", sha.decode())
        if len(data) > 5_000_000:
            continue
        if SECRET.search(data):
            secrets.add(path)
        if USERS.search(data):
            users.add(path)
        if any(m.group() not in ALLOWED_EMAIL for m in EMAIL.finditer(data)):
            emails.add(path)
    print(f"объектов проверено: {len(blobs)}")
    for title, s in (("звук, книги, образцы, веса моделей", files_bad), ("ключи и токены", secrets),
                     ("пути /Users/...", users), ("почтовые адреса", emails)):
        print(f"  {title}: {len(s)}" + (" - " + "; ".join(sorted(s)[:8]) if s else ""))
    found = files_bad or secrets or users or emails
    print("чисто" if not found else "НАЙДЕНО - отправлять нельзя")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
