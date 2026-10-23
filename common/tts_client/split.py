# -*- coding: utf-8 -*-
"""Голос по языку отрывка (описание - README.md рядом): текст - на отрывки по алфавиту, каждый отрывок - голосу своего
языка; голосу никогда не уходит текст чужого алфавита."""
from __future__ import annotations

import re
from typing import Callable, Iterable, Iterator

TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*|[А-Яа-яЁё]+|\s+|[^\sA-Za-zА-Яа-яЁё]+")
CYR = re.compile(r"[А-Яа-яЁё]")
LAT = re.compile(r"[A-Za-z]")


def _lang(tok: str) -> str | None:
    if CYR.search(tok):
        return "ru"
    if LAT.search(tok):
        return "en"
    return None


def segments(text: str, base_lang: str = "ru") -> list[tuple[str, str]]:
    """[(язык, отрывок)] по алфавиту слов (README, 3.1). Текст без букв - один отрывок языка base_lang."""
    out: list[list] = []
    pending = ""
    for tok in TOKEN.findall(text or ""):
        lg = _lang(tok)
        if lg is None:
            if out:
                out[-1][1] += tok
            else:
                pending += tok
            continue
        if out and out[-1][0] == lg:
            out[-1][1] += tok
        else:
            out.append([lg, (pending if not out else "") + tok])
            pending = ""
    res = [(lg, t.strip()) for lg, t in out if t.strip()]
    if not res and (pending or "").strip():
        res = [(base_lang, pending.strip())]
    return res


def assert_clean(text: str, lang: str) -> None:
    """В отрывке - буквы только своего алфавита (README, 3.3)."""
    if (lang == "en" and CYR.search(text)) or (lang == "ru" and LAT.search(text)):
        raise ValueError(f"в отрывке «{text[:40]}» буквы чужого алфавита для голоса {lang}")


def plan(text: str, base_lang: str, voice_for: Callable[[str], str | None]) -> list[tuple[str | None, str, str]]:
    """[(голос или None, язык, отрывок)] - что и каким голосом прозвучит."""
    return [(voice_for(lg), lg, t) for lg, t in segments(text, base_lang)]


def speak(text: str, base_lang: str, voice_for: Callable[[str], str | None],
          synth: Callable[[str, str], Iterable], skipped: list | None = None) -> Iterator:
    """Озвучить текст по отрывкам: synth(отрывок, голос) - звук отрывка (что отдаёт синтез программы)."""
    for voice, lg, t in plan(text, base_lang, voice_for):
        if not voice:
            if skipped is not None:
                skipped.append((lg, t))
            continue
        assert_clean(t, lg)
        yield from synth(t, voice)


__all__ = ["segments", "plan", "speak", "assert_clean"]
