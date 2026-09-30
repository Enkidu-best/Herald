"""Разбивка книги на файлы примерно одинаковой длины.

Раньше здесь искались строки вида «Глава N» и каждая такая строка начинала
новый файл. На реальных книгах это разваливается: у многих книг глав нет вовсе,
заголовок может быть просто выделен жирным, а два заголовка подряд давали файл
на две секунды из одного названия.

Поэтому основной принцип теперь — РАЗМЕР. Текст режется на куски примерно по
`target_minutes` звука. Заголовки не создают файл сами по себе: они лишь
«удобные места для разреза» (лучше закончить файл перед заголовком, чем на
середине абзаца) и дают файлу осмысленное имя.

Здесь только структура и сырой текст. Нормализацию под синтез делает pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .readers import Document, Chapter

# Замерено на готовых файлах: 155 символов дали 13,5 с, 1333 символа — 111 с,
# то есть 12 символов в секунду, а не 14. На звук не влияет — только на прогноз.
CHARS_PER_SEC = 12
DEFAULT_MINUTES = 15.0      # длина одного файла по умолчанию
MIN_FILL = 0.55             # насколько кусок должен быть заполнен, чтобы резать раньше срока
TAIL_MERGE = 0.35           # хвост короче этой доли цели приклеиваем к предыдущему файлу

# Строка похожа на заголовок: «Глава 5», «Часть вторая», «Пролог», «XIV».
_HEADING_WORD = re.compile(
    r"^\s*(?:Глава|ГЛАВА|Часть|ЧАСТЬ|Раздел|РАЗДЕЛ|Книга|Chapter|CHAPTER|Пролог|ПРОЛОГ"
    r"|Эпилог|ЭПИЛОГ|Интерлюдия)\b", re.UNICODE)
_ROMAN_OR_NUM = re.compile(r"^\s*(?:[IVXLCDM]{1,7}|\d{1,3})\s*[.)]?\s*$")


@dataclass
class _Block:
    text: str
    heading: bool


def target_chars(minutes: float) -> int:
    return max(600, int(minutes * 60 * CHARS_PER_SEC))


def build_chapters(doc: Document, target_minutes: float = DEFAULT_MINUTES) -> list[Chapter]:
    """Книга -> список кусков примерно по target_minutes звука каждый."""
    blocks = _to_blocks(doc)
    if not blocks:
        return []
    return _pack(blocks, target_chars(target_minutes))


# --- документ -> плоский список абзацев и заголовков ---------------------
def _to_blocks(doc: Document) -> list[_Block]:
    """Всё содержимое книги одним потоком блоков, порядок сохраняется.

    Структура формата (epub/fb2/docx) здесь не выбрасывается: её заголовки
    становятся блоками-заголовками, то есть предпочтительными местами разреза
    и источником имён. Но границы файлов задаёт размер, а не она.
    """
    blocks: list[_Block] = []
    if doc.chapters:
        for ch in doc.chapters:
            if ch.title and ch.title.strip():
                blocks.append(_Block(ch.title.strip(), True))
            blocks += _paragraphs(ch.text)
    elif doc.text:
        blocks += _paragraphs(doc.text)
    return [b for b in blocks if b.text]


def _paragraphs(text: str) -> list[_Block]:
    out: list[_Block] = []
    for raw in re.split(r"\n\s*\n|\n", text or ""):
        p = raw.strip()
        if p:
            out.append(_Block(p, _looks_like_heading(p)))
    return out


def _looks_like_heading(p: str) -> bool:
    """Короткая отдельная строка без точки на конце — почти наверняка заголовок."""
    if len(p) > 90 or "\n" in p:
        return False
    if _HEADING_WORD.match(p) or _ROMAN_OR_NUM.match(p):
        return True
    # «Открывая пустыню», «Мстя, геноцид и неожиданная прибыль» — тоже заголовки
    return len(p) <= 70 and p[-1] not in ".!?…:,;»\"'" and not p[0].islower()


# --- упаковка блоков в куски нужного размера -----------------------------
def _pack(blocks: list[_Block], target: int) -> list[Chapter]:
    chunks: list[list[_Block]] = []
    cur: list[_Block] = []
    size = 0

    for i, b in enumerate(blocks):
        # заголовок при уже заметно наполненном куске — хорошее место закончить файл
        if b.heading and size >= target * MIN_FILL and _has_body(cur):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(b)
        size += len(b.text)
        # добрали до цели — закрываем, но не разрывая заголовок от его текста
        if size >= target and _has_body(cur) and not _trailing_headings(blocks, i):
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)

    chunks = _merge_tail(chunks, target)
    return [Chapter(_title_for(c, n), _render(c)) for n, c in enumerate(chunks, 1)]


def _has_body(blocks: list[_Block]) -> bool:
    """В куске есть хоть что-то кроме заголовков — иначе это не файл, а пустышка."""
    return any(not b.heading for b in blocks)


def _trailing_headings(blocks: list[_Block], i: int) -> bool:
    """Следующий блок — заголовок? Тогда резать прямо тут и так правильно."""
    return i + 1 < len(blocks) and blocks[i + 1].heading


def _merge_tail(chunks: list[list[_Block]], target: int) -> list[list[_Block]]:
    """Слишком короткий последний кусок приклеиваем к предыдущему.

    Иначе книга заканчивается файлом на полминуты, что выглядит как сбой.
    """
    while len(chunks) > 1:
        tail = sum(len(b.text) for b in chunks[-1])
        if tail >= target * TAIL_MERGE and _has_body(chunks[-1]):
            break
        chunks[-2] = chunks[-2] + chunks.pop()
    return chunks


# Служебные пометки магазинов и библиотек в названии: «Жажда власти [litres]».
# В имени папки и в тегах они только мешают.
_VENDOR = re.compile(r"\s*[\[(](?:litres|литрес|litmir|литмир|flibusta|флибуста"
                     r"|hardcase|fb2|epub)[\])]\s*", re.I)


def clean_book_title(t: str) -> str:
    """Название книги без «[litres]» и подобного, без лишних пробелов."""
    t = _VENDOR.sub(" ", t or "")
    return re.sub(r"\s+", " ", t).strip(" .—-_") or (t or "").strip()


def part_name(n: int, total: int) -> str:
    """Имя файла: всегда «Часть N», единообразно для любой книги.

    Номер дополняется нулём до ширины общего числа частей («Часть 01» … «Часть
    12»): иначе плееры, сортирующие по буквам, ставят «Часть 10» перед «Часть 2».
    Настоящий заголовок главы идёт в теги файла, а не в имя.
    """
    return f"Часть {n:0{len(str(max(total, 1)))}d}"


def _title_for(blocks: list[_Block], n: int) -> str:
    for b in blocks:
        if b.heading:
            return re.sub(r"\s+", " ", b.text).strip()[:70]
    return f"Часть {n}"


def _render(blocks: list[_Block]) -> str:
    return "\n\n".join(b.text for b in blocks)
