"""Подготовка русского текста к синтезу.

Это главный рычаг качества звучания. Движок читает то, что ему дали: «1990 г.»
он произнесёт как «одна тысяча девятьсот девяносто г», если не развернуть числа
и сокращения в слова заранее. Здесь мы это и делаем, плюс чистим мусор от PDF
и переносы.

Функции написаны консервативно: лучше не тронуть спорное, чем испортить текст.
"""

from __future__ import annotations

import re
import unicodedata

try:
    from num2words import num2words
except Exception:  # пакет может быть не установлен в среде без синтеза
    num2words = None


# --- сокращения: только однозначные, чтобы не наломать -------------------
_ABBR = [
    (r"\bт\.\s*е\.", "то есть"),
    (r"\bт\.\s*к\.", "так как"),
    (r"\bт\.\s*д\.", "так далее"),
    (r"\bт\.\s*п\.", "тому подобное"),
    (r"\bт\.\s*н\.", "так называемый"),
    (r"\bи\s+др\.", "и другие"),
    (r"\bи\s+пр\.", "и прочее"),
    (r"\bсм\.", "смотри"),
    (r"\bстр\.", "страница"),
    (r"\bрис\.", "рисунок"),
    (r"\bтаб\.", "таблица"),
    (r"\bн\.\s*э\.", "нашей эры"),
    (r"\bдо\s+н\.\s*э\.", "до нашей эры"),
    (r"\bгг\.", "годы"),
    (r"\bтыс\.", "тысяч"),
    (r"\bмлн\b\.?", "миллионов"),
    (r"\bмлрд\b\.?", "миллиардов"),
    (r"\bруб\.", "рублей"),
    (r"\bкоп\.", "копеек"),
]

_SYMBOLS = [
    ("«", ""), ("»", ""), ("“", ""), ("”", ""), ("„", ""),
    # Скобки модель не «слышит» как паузу и склеивает соседние слова:
    # «от глаз (или, скорее…» читалось как «от глазили». Меняем на запятые —
    # получается ровно та интонационная вставка, которой скобки и являются.
    (" (", ", "), ("(", ", "), (")", ","),
    (" [", ", "), ("[", ", "), ("]", ","),
    ("…", "..."),
    ("—", " - "), ("–", " - "), ("‑", "-"),
    (" ", " "), (" ", " "), (" ", " "),
    ("№", " номер "), ("§", " параграф "),
    ("°", " градусов"),
    ("&", " и "),
]


def _strip_control(text: str) -> str:
    out = []
    for ch in text:
        if ch in "\n\t":
            out.append(ch)
        elif unicodedata.category(ch)[0] == "C":   # прочие управляющие
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _dehyphenate(text: str) -> str:
    """Склеить перенос слова через дефис в конце строки: «соеди-\\nнение»."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _numbers_to_words(text: str) -> str:
    if num2words is None:
        return text

    # убрать разделители тысяч между цифрами: «1 000 000» -> «1000000»
    text = re.sub(r"(?<=\d)[  ](?=\d{3}\b)", "", text)

    def repl_decimal(m: re.Match) -> str:
        whole, frac = m.group(1), m.group(2)
        try:
            return num2words(float(f"{whole}.{frac}"), lang="ru")
        except Exception:
            return m.group(0)

    # десятичные с запятой: «3,5»
    text = re.sub(r"\b(\d+),(\d+)\b", repl_decimal, text)

    def repl_int(m: re.Match) -> str:
        try:
            return num2words(int(m.group(0)), lang="ru")
        except Exception:
            return m.group(0)

    # целые числа (в т.ч. годы — как количественные, этого для v1 достаточно)
    text = re.sub(r"\b\d+\b", repl_int, text)
    return text


# --- порядковые числительные, годы, римские цифры ------------------------
# Окончания по падежу. Ключ — то, что стоит в тексте после дефиса («5-го»),
# значение — окончание для твёрдого и мягкого склонения («второго», «третьего»).
_ORD_ENDINGS = {
    "й": ("ый", "ий"), "ый": ("ый", "ий"), "ой": ("ый", "ий"), "ий": ("ый", "ий"),
    "го": ("ого", "ьего"), "ого": ("ого", "ьего"), "его": ("ого", "ьего"),
    "му": ("ому", "ьему"), "ому": ("ому", "ьему"),
    "м": ("ом", "ьем"), "ом": ("ом", "ьем"), "ем": ("ом", "ьем"),
    "е": ("ое", "ье"), "ое": ("ое", "ье"), "ее": ("ое", "ье"),
    "я": ("ая", "ья"), "ая": ("ая", "ья"), "яя": ("ая", "ья"),
    "ю": ("ую", "ью"), "ую": ("ую", "ью"), "юю": ("ую", "ью"),
    "ых": ("ых", "ьих"), "х": ("ых", "ьих"), "их": ("ых", "ьих"),
    "ые": ("ые", "ьи"), "ие": ("ые", "ьи"),
    "ыми": ("ыми", "ьими"), "ми": ("ыми", "ьими"), "ими": ("ыми", "ьими"),
}

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
# слова, рядом с которыми римская цифра точно цифра, а не буква
_ROMAN_CONTEXT_F = r"(?:глав|част|книг|сери)"        # женский род: «глава четырнадцатая»
_ROMAN_CONTEXT_M = r"(?:том|раздел|век|акт|этап|уровен|класс)"


def _ordinal_form(n: int, suffix: str, feminine: bool = False) -> str:
    """Порядковое числительное в нужной форме: (5, "е") -> «пятое»."""
    if num2words is None:
        return str(n)
    base_word = num2words(n, lang="ru", to="ordinal")
    head, _, last = base_word.rpartition(" ")
    soft = last.endswith("ий")              # «третий» склоняется мягко
    stem = last[:-2] if last[-2:] in ("ый", "ой", "ий") else last
    hard_end, soft_end = _ORD_ENDINGS.get(suffix, ("ый", "ий"))
    if feminine:
        hard_end, soft_end = ("ая", "ья")
    word = stem + (soft_end if soft else hard_end)
    return f"{head} {word}".strip()


def _roman_to_int(s: str) -> int | None:
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _ROMAN_VALUES.get(ch)
        if v is None:
            return None
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total or None


def _expand_ordinals(text: str) -> str:
    """«5-е» -> «пятое», «2-го» -> «второго». Без этого выходило «пять-е»."""
    def repl(m):
        n, suf = int(m.group(1)), m.group(2).lower()
        return _ordinal_form(n, suf)
    return re.sub(r"\b(\d+)-(ыми|ими|ими|ого|его|ому|ему|ой|ый|ий|ая|яя|ую|юю|ое|ее|ые|ие|ых|их|ом|ем|го|му|ми|й|я|е|ю|м|х)\b",
                  repl, text, flags=re.IGNORECASE)


def _expand_years(text: str) -> str:
    """«в 1990 году» -> «в тысяча девятьсот девяностом году» (а не «девяносто году»)."""
    cases = {"год": "й", "года": "го", "году": "м", "годом": "ым", "годе": "м"}

    def repl(m):
        n, word = int(m.group(1)), m.group(2).lower()
        return f"{_ordinal_form(n, cases.get(word, 'й'))} {m.group(2)}"
    return re.sub(r"\b(\d{3,4})\s+(год|года|году|годом|годе)\b", repl, text,
                  flags=re.IGNORECASE)


def _expand_roman(text: str) -> str:
    """«Глава XIV» -> «Глава четырнадцатая», «XIX век» -> «девятнадцатый век».

    Только рядом с опорными словами: иначе одиночные I, V, X, C — это буквы,
    и превращать их в числа опаснее, чем оставить.
    """
    def repl_after(m):
        n = _roman_to_int(m.group(2))
        if not n or n > 200:
            return m.group(0)
        fem = bool(re.match(_ROMAN_CONTEXT_F, m.group(1), re.IGNORECASE))
        return f"{m.group(1)} {_ordinal_form(n, 'й', feminine=fem)}"

    def repl_before(m):
        n = _roman_to_int(m.group(1))
        if not n or n > 200:
            return m.group(0)
        return f"{_ordinal_form(n, 'й')} {m.group(2)}"

    text = re.sub(rf"\b({_ROMAN_CONTEXT_F}\w*|{_ROMAN_CONTEXT_M}\w*)\s+([IVXLCDM]{{1,7}})\b",
                  repl_after, text, flags=re.IGNORECASE)
    text = re.sub(rf"\b([IVXLCDM]{{2,7}})\s+({_ROMAN_CONTEXT_M}\w*)\b",
                  repl_before, text, flags=re.IGNORECASE)
    return text


def _expand_percent(text: str) -> str:
    """«1%» -> «один процент», «5%» -> «пять процентов» (раньше был «один процентов»)."""
    def repl(m):
        n = int(m.group(1))
        last, last2 = n % 10, n % 100
        if last == 1 and last2 != 11:
            word = "процент"
        elif last in (2, 3, 4) and last2 not in (12, 13, 14):
            word = "процента"
        else:
            word = "процентов"
        return f"{m.group(1)} {word}"
    text = re.sub(r"\b(\d+)\s*%", repl, text)
    return text.replace("%", " процентов")


def normalize_text(text: str, expand_numbers: bool = True) -> str:
    """Главная функция: сырой фрагмент -> текст, готовый к синтезу."""
    if not text:
        return ""
    text = _strip_control(text)
    text = _dehyphenate(text)

    for a, b in _SYMBOLS:
        text = text.replace(a, b)

    for pat, rep in _ABBR:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # «1990 г.» -> «1990 года» (4 цифры почти всегда год, а не граммы)
    text = re.sub(r"(\d{4})\s*г\.", r"\1 года", text)

    if expand_numbers:
        # порядок важен: сначала то, где цифра читается особым образом
        # (порядковые, годы, проценты), и только потом — обычные числа
        text = _expand_roman(text)
        text = _expand_ordinals(text)
        text = _expand_years(text)
        text = _expand_percent(text)
        text = _numbers_to_words(text)

    # после замены скобок бывает «,,» и «, .» — подчищаем
    text = re.sub(r",\s*([,.!?;:…])", r"\1", text)
    text = re.sub(r",{2,}", ",", text)

    # одиночные переводы строки -> пробел (абзацы разделяются пустой строкой)
    text = re.sub(r"[ \t]*\n[ \t]*\n[ \t]*", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --- разбивка на предложения --------------------------------------------
def sentenize(text: str) -> list[str]:
    """Список предложений. Через razdel, с запасным вариантом по знакам."""
    text = text.strip()
    if not text:
        return []
    try:
        from razdel import sentenize as _rz
        return [s.text.strip() for s in _rz(text) if s.text.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?…])\s+", text)
        return [p.strip() for p in parts if p.strip()]
