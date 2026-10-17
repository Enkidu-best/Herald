#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-TTS 1.7B ru-stress-cf -> MLX, 8 и 4 бита (3.8, задание 10а, п. 1.2).

Запускать окружением mlx-audio (тяжёлый шаг - под ../.heavy-lock: скачивание
~4,5 ГБ и конвертация):

    .venv-mlxaudio/bin/python "Инструменты разработки/convert_qwen_stress.py" [--bits 8 4]

Модель кладётся в папку данных Herald: ~/.cache/text2audio/models/
qwen3-tts-ru-stress-{8,4}bit. Если конвертация не идёт - причина печатается и
пишется в models/qwen3-tts-ru-stress.log (запасные пути - GGUF-сборка
sknyazev/qwen3-tts-12hz-1.7b-ru-stress-gguf или PyTorch/MPS, см. Приёмку 3.8).
"""
import argparse
import os
import subprocess
import sys
import time
import traceback

REPO = "siriusfreak/qwen3-tts-12hz-1.7b-ru-stress-cf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/text2audio/models"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "qwen3-tts-ru-stress.log"), "a", encoding="utf-8")
    rc = 0
    for bits in a.bits:
        dst = os.path.join(a.out, f"qwen3-tts-ru-stress-{bits}bit")
        if os.path.exists(os.path.join(dst, "config.json")):
            print(f"{bits} бит: уже есть - {dst}")
            continue
        cmd = [sys.executable, "-m", "mlx_audio.convert", "--hf-path", REPO,
               "--mlx-path", dst, "-q", "--q-bits", str(bits), "--model-domain", "tts"]
        print(" ".join(cmd), flush=True)
        t0 = time.monotonic()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            took = time.monotonic() - t0
            ok = r.returncode == 0 and os.path.exists(os.path.join(dst, "config.json"))
            msg = f"{time.ctime()} {bits} бит: {'готово' if ok else 'НЕ ВЫШЛО'} за {took:.0f} с"
            print(msg, flush=True)
            log.write(msg + "\n")
            if not ok:
                rc = 1
                tail = (r.stderr or r.stdout)[-3000:]
                print(tail)
                log.write(tail + "\n")
        except Exception:
            rc = 1
            log.write(traceback.format_exc() + "\n")
            traceback.print_exc()
    return rc


if __name__ == "__main__":
    sys.exit(main())
