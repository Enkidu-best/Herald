#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Сквозная проверка Herald на Windows и Linux (раннеры GitHub, Docker).

Всё через консольную программу - так одинаково проверяются и исходники
(`python cli.py`), и установленный `herald-cli.exe` (--exe):
  1. чем считается озвучка (--engine-info): на Mac - MLX, иначе F5 на torch;
  2. пробник готовым голосом Silero (русский) - быстро и на процессоре;
  3. из этого пробника - образец голоса (--ref): отбор отрывка и расшифровка;
  4. пробник КЛОНОМ: F5 на torch, 12 шагов, якорь тишины, проверка слов;
  5. распознавание (--asr): слышно ли то, что написано.
Образцов чужих голосов здесь нет: голос для клона делается тут же из Silero.

    python "Инструменты разработки/smoke_pc.py" [--exe путь\\herald-cli.exe] [--quick]
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.path.join(ROOT, "Примеры текстов", "Короткий тест - ru.txt")
fails: list[str] = []


def must(ok, msg):
    print(("  ✓ " if ok else "  ✗ ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def words(s):
    return re.sub(r"[^а-яa-z ]", " ", s.lower().replace("ё", "е")).split()


def run(cli, *args, timeout=3600):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run(cli + list(args), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=env, cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:], sep="\n")
    return r, dt


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=None)
    ap.add_argument("--quick", action="store_true", help="без клона F5 (только Silero)")
    a = ap.parse_args()
    cli = [a.exe] if a.exe else [sys.executable, os.path.join(ROOT, "cli.py")]
    out = tempfile.mkdtemp(prefix="herald_smoke_")
    report = {}

    print("1. Чем считается:")
    r, _ = run(cli, "--engine-info")
    info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
    print("    ", info)
    must(info.get("engine") in ("f5torch", "f5mlx"), f"движок клона: {info.get('engine')}")
    report["engine"] = info

    print("2. Пробник Silero:")
    r, dt = run(cli, TEXT, "--engine", "silero", "--voice", "aidar", "--preview", "1",
                "--out", out)
    sil = os.path.join(out, "Короткий тест - ru - пробник.mp3")
    d = duration(sil) if os.path.exists(sil) else 0
    must(r.returncode == 0 and d > 5, f"Silero: {d:.1f} с звука за {dt:.0f} с")
    report["silero"] = {"sec": round(d, 1), "took": round(dt)}
    if a.quick or not d:
        return finish(report)

    print("3. Образец голоса из пробника Silero и 4. пробник клоном:")
    clone_dir = os.path.join(out, "клон")
    os.makedirs(clone_dir, exist_ok=True)
    r, dt = run(cli, TEXT, "--engine", "auto", "--ref", sil, "--preview", "1",
                "--out", clone_dir, timeout=4 * 3600)
    mp3 = os.path.join(clone_dir, "Короткий тест - ru - пробник.mp3")
    d = duration(mp3) if os.path.exists(mp3) else 0
    must(r.returncode == 0 and d > 5, f"клон F5: {d:.1f} с звука за {dt:.0f} с (RTF {dt / max(d, 1):.1f})")
    report["clone"] = {"sec": round(d, 1), "took": round(dt), "rtf": round(dt / max(d, 1), 1)}

    print("5. Распознавание клона:")
    if d:
        r, _ = run(cli, "--asr", mp3)
        heard = json.loads(r.stdout.strip().splitlines()[-1])["text"] if r.returncode == 0 else ""
        want = words(open(TEXT, encoding="utf-8").read())
        got = words(heard)
        m = difflib.SequenceMatcher(None, want, got)
        share = sum(b.size for b in m.get_matching_blocks()) / max(1, len(want))
        print("     услышано:", heard[:200])
        must(share >= 0.6, f"слов совпало {share:.0%}")
        report["clone"]["words"] = round(share, 3)
    return finish(report)


def finish(report) -> int:
    print(json.dumps(report, ensure_ascii=False))
    path = os.environ.get("HERALD_SMOKE_REPORT")
    if path:
        json.dump(report, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nOK" if not fails else f"\nПРОВАЛ: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
