#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Словарь по Зализняку (DataverseNO, doi:10.18710/AAFCJP, CC0 1.0; Schriner 2022, из
«Грамматического словаря» 1977 через Giella) -> core/data/stress_zaliznyak.tsv.gz:
форма (е вместо ё) \t номера ударных гласных (основное ударение, через запятую варианты).

    python "Инструменты разработки/zaliznyak_to_tsv.py" Schriner-2022-RU_Zaliznyak.tsv
"""
import gzip
import sys

VOW = "аеёиоуыэюя"


def main(src: str, dst: str = "core/data/stress_zaliznyak.tsv.gz") -> int:
    out: dict[str, set[int]] = {}
    for line in open(src, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 2 or p[0] == "word_without_stress_markers":
            continue
        st = p[1].lower()
        vi, idx = -1, None
        for i, c in enumerate(st):
            if c in VOW:
                vi += 1
                if c == "ё" or (i + 1 < len(st) and st[i + 1] == "́"):
                    idx = vi
        if idx is not None:
            out.setdefault(p[0].lower().replace("ё", "е"), set()).add(idx)
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write("# Ударения по «Грамматическому словарю» А. А. Зализняка (1977): DataverseNO "
                "doi:10.18710/AAFCJP, CC0 1.0. форма\tномера ударных гласных\n")
        for k in sorted(out):
            f.write(f"{k}\t{','.join(map(str, sorted(out[k])))}\n")
    print("форм:", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
