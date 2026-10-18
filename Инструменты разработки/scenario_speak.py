#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сценарий 3.8: /speak для Interview Coach работает как раньше.

Interview Coach (coach/herald.py, coach/tts.py) пользуется Herald так:
  * запускает его Python: `cli.py --serve --port <порт>` (порт - speak_port из
    ~/.cache/text2audio/herald.json, иначе 8716), ждёт GET /health = 200;
  * POST /speak {"text", "fast": true, "voice": "<id Silero>"} -> поток WAV:
    заголовок 44 байта, 48 кГц, int16, моно;
  * каталог быстрых голосов читает текстом из core/voice_store.py своим
    регулярным выражением (FastVoice("id", "имя", "пол", "язык", "движок")).
Здесь - ровно это, на ОТДЕЛЬНОМ сервере из текущего кода на свободном порту:
работающий сервер Interview Coach не трогается (у него только спрашивается
/health). Тяжёлый шаг (Silero) - под ../.heavy-lock.

    python "Инструменты разработки/scenario_speak.py"
"""
from __future__ import annotations

import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
# то же выражение, что в Interview Coach (coach/herald.py, _FV)
FV = re.compile(r'FastVoice\(\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]*)",\s*"(\w\w)",\s*"(\w+)"')
FAILED: list[str] = []


def must(cond, msg):
    print(("  OK    " if cond else "  ПАДАЕТ ") + msg, flush=True)
    if not cond:
        FAILED.append(msg)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def health(port: int, timeout=0.8) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def speak(port: int, text: str, voice: str):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/speak", method="POST",
                                 data=json.dumps({"text": text, "fast": True, "voice": voice}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    data = b""
    with urllib.request.urlopen(req, timeout=60) as r:
        status = r.status
        while True:
            b = r.read(9600)
            if not b:
                break
            if first is None:
                first = time.monotonic() - t0
            data += b
    return status, data, first, time.monotonic() - t0


def main() -> int:
    print("каталог голосов - как его читает Interview Coach:")
    cat = FV.findall(open(os.path.join(ROOT, "core", "voice_store.py"), encoding="utf-8").read())
    ids = [c[0] for c in cat]
    must({"aidar", "eugene", "kseniya", "xenia"} <= set(ids) and all(c[4] in ("silero", "vosk", "piper")
                                                                    for c in cat),
         f"FastVoice читаются: {len(cat)} голосов ({', '.join(ids[:5])}…)")
    try:
        cfg = json.load(open(os.path.expanduser("~/.cache/text2audio/herald.json"), encoding="utf-8"))
        ic_port = int(cfg.get("speak_port") or 8716)
        hidden = cfg.get("hidden_voices")
        must(hidden is None or isinstance(hidden, list), "hidden_voices в herald.json - список")
    except (OSError, ValueError):
        ic_port = 8716
    running = [p for p in {ic_port, 61397} if health(p)]
    print(f"работающий сервер Interview Coach: {running or 'нет'} (не трогаем)")
    print("отдельный сервер из текущего кода:")
    port = free_port()
    proc = subprocess.Popen([PY, "cli.py", "--serve", "--port", str(port)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60 and not health(port):
            time.sleep(0.3)
        must(health(port), f"GET /health = 200 через {time.monotonic() - t0:.1f} с")
        for voice, text in (("aidar", "Расскажите, пожалуйста, о себе."),
                            ("kseniya", "Спасибо, следующий вопрос."),
                            ("xenia", "Хорошо. А почему вы ушли с прошлой работы?")):
            status, data, first, total = speak(port, text, voice)
            ok_head = len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
            sr = struct.unpack("<I", data[24:28])[0] if ok_head else 0
            ch = struct.unpack("<H", data[22:24])[0] if ok_head else 0
            bits = struct.unpack("<H", data[34:36])[0] if ok_head else 0
            sec = (len(data) - 44) / 2 / 48000
            must(status == 200 and ok_head and sr == 48000 and ch == 1 and bits == 16 and sec > 0.8,
                 f"/speak {voice}: WAV 48 кГц int16 моно, {sec:.1f} с звука, первый звук "
                 f"{first:.2f} с, всего {total:.2f} с")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    for p in running:
        must(health(p), f"сервер Interview Coach (порт {p}) жив и после проверки")
    print("всё в порядке" if not FAILED else f"\nПРОВАЛЕНО: {len(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
