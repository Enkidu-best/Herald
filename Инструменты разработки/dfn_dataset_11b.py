#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 11б: набор фраз -> «восстановленный» DeepFilterNet 3 (как у Ященко в точке В2; Sidon
меняет голос - шаг 2). Окружение .venv-restore (soundfile, librosa) и программа deep-filter.

    .venv-restore/bin/python "Инструменты разработки/dfn_dataset_11b.py" <чтец> <вход.jsonl> <выход.jsonl> <папка wav>
"""
import json, os, subprocess, sys, tempfile, time
import librosa, soundfile as sf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DFN = os.path.join(HERE, ".venv-restore", "bin-dfn", "deep-filter-0.5.6-aarch64-apple-darwin")


def main() -> int:
    voice, src, dst, sub = sys.argv[1:5]
    d = os.path.expanduser(f"~/.cache/text2audio/train/{voice}")
    out = os.path.join(d, sub)
    os.makedirs(out, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(d, src), encoding="utf-8")]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        ins = []
        for k, r in enumerate(rows):
            a, sr = sf.read(r["audio"], dtype="float32")
            p = os.path.join(tmp, f"{k:05d}.wav")
            sf.write(p, librosa.resample(a, orig_sr=sr, target_sr=48000), 48000, subtype="PCM_16")
            ins.append(p)
        os.makedirs(os.path.join(tmp, "o"))
        for i in range(0, len(ins), 200):
            subprocess.run([DFN, "-D", "-o", os.path.join(tmp, "o"), *ins[i:i + 200]], check=True, capture_output=True)
        with open(os.path.join(d, dst), "w", encoding="utf-8") as f:
            for k, r in enumerate(rows):
                b, sr = sf.read(os.path.join(tmp, "o", f"{k:05d}.wav"), dtype="float32")
                p = os.path.join(out, os.path.basename(r["audio"]))
                sf.write(p, librosa.resample(b, orig_sr=sr, target_sr=24000), 24000)
                f.write(json.dumps(dict(r, audio=p), ensure_ascii=False) + "\n")
    print(f"{voice}: {len(rows)} фраз через DeepFilterNet за {time.time() - t0:.0f} с", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
