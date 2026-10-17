"""Расстановка ударений в русском тексте: RUAccent + silero-stress.

Схема выбрана замером на открытой разметке (Викисловарь, 9 141 предложение с
ударениями, см. «Инструменты разработки/bench_stress.py» и docs/Отчёт этапа 1):

                         RUAccent turbo3.1   silero-stress 1.5
    обычные слова             99,4 %*             99,2 %
    омографы                  90,1 %              91,9 %
    (* из размеченных: служебные «или», «это», «было» RUAccent нарочно
       оставляет без ударения - так, по-видимому, и размечены данные, на которых
       дообучали модель синтеза)

Поэтому основа - RUAccent, а в омографах (за́мок/замо́к, сте́ны/стены́, все/всё)
берётся выбор silero-stress. Если silero-stress не загрузился - работает один
RUAccent, как раньше.

Отдельно - единообразие по книге: слово вне словаря (имя героя, выдуманное
слово) должно звучать одинаково во всей книге, а расстановщик угадывает его
заново в каждом месте (в «Жажде власти» «ланд-капитан» шёл 18 раз с ударением
на «ланд» и 11 - на «капитан»). unify_book() выбирает для такого слова самый
частый вариант и ставит его везде. Омографы и словарные слова не трогаются.

Ставит ударение автоматически на любой текст - это масштабируемо, ничего не
прописывается вручную. Формат вывода - '+' перед ударной гласной («молок+о»),
именно его понимают Silero и русская модель F5 (accent_tune).

Модуль устойчив к сбоям: если RUAccent недоступен или падает, возвращаем текст
как есть (лучше без ударений, чем уронить всю начитку).
"""

from __future__ import annotations

import os
import re
import threading

from .log import log_problem

_LOCK = threading.Lock()
_ACCENTIZER = None
_TRIED = False


def _patch_onnx_inputs(acc) -> None:
    """Дополнить недостающие входы ONNX-моделей RUAccent нулями.

    С transformers 5.x токенизатор RUAccent перестал отдавать token_type_ids, а
    ONNX-модель, ставящая ударение в словах вне словаря, требует их. Модель
    падала, и вместе с ней - вся фраза: add_stress возвращал её совсем БЕЗ
    ударений. На фантастике это каждая фраза с именем героя или выдуманным
    словом. token_type_ids для одиночного текста - это одни нули, так и было до
    обновления, поэтому дополнение нулями ничего в модели не меняет.
    """
    import numpy as np
    for attr in ("accent_model", "omograph_model", "stress_usage_predictor",
                 "yo_homograph_model"):
        model = getattr(acc, attr, None)
        sess = getattr(model, "session", None)
        if sess is None or getattr(sess, "_herald_patched", False):
            continue
        need = [i.name for i in sess.get_inputs()]
        if "token_type_ids" not in need:
            continue
        run = sess.run

        def patched(output_names, feed, *a, _run=run, **kw):
            if "token_type_ids" not in feed and "input_ids" in feed:
                feed = dict(feed, token_type_ids=np.zeros_like(feed["input_ids"]))
            return _run(output_names, feed, *a, **kw)

        try:
            sess.run = patched
            sess._herald_patched = True
        except Exception:
            # у некоторых сборок onnxruntime атрибуты сессии только для чтения —
            # тогда оборачиваем саму модель
            model.session = _SessionProxy(sess, patched)


class _SessionProxy:
    def __init__(self, sess, run):
        self._sess, self.run, self._herald_patched = sess, run, True

    def __getattr__(self, name):
        return getattr(self._sess, name)


def _get_accentizer():
    """Лениво загрузить RUAccent один раз. При любой ошибке вернуть None."""
    global _ACCENTIZER, _TRIED
    if _ACCENTIZER is not None or _TRIED:
        return _ACCENTIZER
    with _LOCK:
        if _ACCENTIZER is not None or _TRIED:
            return _ACCENTIZER
        _TRIED = True
        try:
            from ruaccent import RUAccent
            acc = RUAccent()
            # по убыванию качества; берём первый, который загрузится
            for kw in (
                dict(omograph_model_size="turbo3.1", use_dictionary=True),
                dict(omograph_model_size="turbo", use_dictionary=True),
                dict(tiny_mode=True, use_dictionary=True),
                dict(),
            ):
                try:
                    acc.load(**kw)
                    _patch_onnx_inputs(acc)
                    _ACCENTIZER = acc
                    break
                except Exception as e:
                    log_problem(f"RUAccent: не загрузилась конфигурация {kw}", e)
                    continue
        except Exception:
            _ACCENTIZER = None
    return _ACCENTIZER


def add_stress(text: str) -> str:
    """Вернуть текст с ударениями ('+' перед ударной гласной). При сбое - как есть."""
    if not text or not text.strip():
        return text
    acc = _get_accentizer()
    if acc is None:
        return text
    try:
        return _fix_dialog_dash(acc.process_all(text))
    except Exception as e:
        # RUAccent может спотыкаться на отдельных фрагментах - не роняем
        # начитку, но и молча не глотаем: фраза без ударений звучит заметно
        # хуже, и раньше именно так месяцами терялись ударения во всех фразах с
        # именами (см. _patch_onnx_inputs)
        log_problem("RUAccent упал на тексте, ставлю ударения по фразам", e)
        out, lost = [], 0
        for part in re.split(r"(?<=[.!?…])\s+", text):
            try:
                out.append(acc.process_all(part))
            except Exception as e2:
                lost += 1
                out.append(part)
                if lost == 1:
                    log_problem(f"фраза осталась без ударений: «{part[:60]}»", e2)
        if lost:
            log_problem(f"без ударений осталось фраз: {lost}", None, show=True)
        return _fix_dialog_dash(" ".join(out))


def _fix_dialog_dash(text: str) -> str:
    """Вернуть пробел после тире в начале реплики.

    RUAccent склеивает «- Ты правда…» в «-Ты правда…»: тире прилипает к слову, и
    модель синтеза видит уже не реплику, а слово с дефисом. Слова с дефиса не
    начинаются, поэтому правило общее и безопасное.
    """
    return re.sub(r"(^|\n)([ \t]*)-(?=[^\s\-])", r"\1\2- ", text)


# --- silero-stress: омографы --------------------------------------------
_SILERO = None
_SILERO_TRIED = False
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
_WORD = re.compile(r"[А-Яа-яЁё+]+(?:-[А-Яа-яЁё+]+)*")


def _get_silero():
    global _SILERO, _SILERO_TRIED
    if _SILERO is not None or _SILERO_TRIED:
        return _SILERO
    with _LOCK:
        if _SILERO is not None or _SILERO_TRIED:
            return _SILERO
        _SILERO_TRIED = True
        try:
            from silero_stress import load_accentor
            _SILERO = load_accentor()
        except Exception as e:
            log_problem("silero-stress не загрузился: омографы будут по RUAccent", e)
            _SILERO = None
    return _SILERO


def _key(w: str) -> str:
    return w.replace("+", "").lower().replace("ё", "е")


def _stress_idx(w: str) -> tuple:
    """Номера ударных гласных слова: по «+», а без «+» - по «ё»."""
    out, vi, plus, yo = [], -1, False, None
    for c in w:
        if c == "+":
            plus = True
            continue
        if c in _VOWELS:
            vi += 1
            if plus:
                out.append(vi)
            elif c in "ёЁ" and yo is None:
                yo = vi
        plus = False
    return tuple(out) if out else ((yo,) if yo is not None else ())


def _apply_idx(w: str, idx: tuple) -> str:
    plain, out, vi = w.replace("+", ""), [], -1
    for c in plain:
        if c in _VOWELS:
            vi += 1
            if vi in idx:
                out.append("+")
        out.append(c)
    return "".join(out)


def _silero_text(text: str) -> str:
    si = _get_silero()
    parts = re.split(r"((?<=[.!?…])\s+|\n+)", text)
    out = []
    for p in parts:
        if not p or not p.strip() or not re.search("[А-Яа-яЁё]", p):
            out.append(p)
            continue
        try:
            out.append(si(p, stress_single_vowel=False))
        except Exception as e:
            log_problem(f"silero-stress не разобрал фразу «{p[:50]}»", e, show=False)
            out.append(p)
    return "".join(out)


# Уверенность RUAccent в выборе омографа (3.7). Сам RUAccent её не отдаёт -
# выбор омографа (omograph_model.classify) заменён копией с той же логикой,
# которая заодно запоминает вероятность выбранного варианта для каждого слова.
_CONF: dict[str, float] = {}
# Ниже этой уверенности RUAccent омограф берётся у silero-stress. Подбор 3.7
# (Инструменты разработки/bench_stress_stages.py): 30 трудных фраз - 27 верно
# (silero всегда - 24, один RUAccent - 24), 49 омографов - 44 (43 и 43)
SILERO_BELOW = 0.8


def _wrap_classify(acc) -> None:
    om = getattr(acc, "omograph_model", None)
    if om is None or getattr(om, "_herald_conf", False):
        return
    import numpy as np

    def classify(texts, hypotheses, num_hypotheses):
        outs, pos = [], 0
        for n in num_hypotheses:
            t = re.sub(r"\s+(?=(?:[,.?!:;…]))", "", texts[pos])
            hs = hypotheses[pos:pos + n]
            probs = []
            for h in hs:
                inp = om.tokenizer(t, h, max_length=512, truncation=True, return_tensors="np")
                inp = {k: v.astype(np.int64) for k, v in inp.items()}
                o = om.softmax(om.session.run(None, inp)[0])
                probs.append(float(o[0][1]))
            best = int(np.argmax(probs))
            tot = sum(probs) or 1.0
            k = _key(hs[best])
            _CONF[k] = min(_CONF.get(k, 1.0), probs[best] / tot)
            outs.append(hs[best])
            pos += n
        return outs
    om.classify = classify
    om._herald_conf = True


def _homographs_from_silero(ru_out: str, src: str) -> str:
    """Омографы: выбор RUAccent; если он не уверен (ниже SILERO_BELOW) - silero.

    До 3.7 silero-stress перебивал ВСЕ омографы - и портил «взял+ись», «глаз+а»,
    «в+олны», которые RUAccent ставил верно (жалоба Андрея на «Пробу пера»)."""
    import difflib
    acc = _get_accentizer()
    if _get_silero() is None or acc is None:
        return ru_out
    homos = acc.omographs
    yo_homos = getattr(acc, "yo_homographs", {})
    si_out = _silero_text(src)
    rm = list(_WORD.finditer(ru_out))
    sw = _WORD.findall(si_out)
    sm = difflib.SequenceMatcher(None, [_key(m.group()) for m in rm],
                                 [_key(w) for w in sw], autojunk=False)
    repl = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for m, w in zip(rm[i1:i2], sw[j1:j2]):
            k = _key(m.group())
            if ((k in homos or k in yo_homos) and _stress_idx(m.group()) != _stress_idx(w)
                    and _CONF.get(k, 0.0) < SILERO_BELOW):
                repl.append((m.start(), m.end(), w))
    for a, b, w in reversed(repl):
        ru_out = ru_out[:a] + w + ru_out[b:]
    return ru_out


# --- ударения пользователя (3.7) -----------------------------------------
# Андрей дописывает слово с ударением («взял+ись») - оно важнее RUAccent. Файл -
# рядом с настройками (папка программы после установки закрыта на запись).
USER_DICT = os.path.join(os.path.expanduser("~"), ".cache", "text2audio", "dicts",
                         "ударения_пользователя.txt")
_user_cache: tuple[float, dict] | None = None


def user_stress() -> dict[str, str]:
    """{слово без ударения: слово с «+»} из файла пользователя (перечитывается,
    если файл изменился)."""
    global _user_cache
    try:
        mt = os.path.getmtime(USER_DICT)
    except OSError:
        return {}
    if _user_cache is None or _user_cache[0] != mt:
        d = {}
        for line in open(USER_DICT, encoding="utf-8"):
            w = line.split("#", 1)[0].strip()
            if "+" in w and _WORD.fullmatch(w):
                d[_key(w)] = w.lower()
        _user_cache = (mt, d)
    return _user_cache[1]


def set_user_stress(word: str) -> str:
    """Добавить или заменить слово с ударением в словаре пользователя."""
    w = word.strip().lower()
    if "+" not in w or not _WORD.fullmatch(w):
        raise ValueError("нужно слово с «+» перед ударной гласной, например «взял+ись»")
    os.makedirs(os.path.dirname(USER_DICT), exist_ok=True)
    lines = []
    if os.path.exists(USER_DICT):
        lines = [x for x in open(USER_DICT, encoding="utf-8").read().splitlines()
                 if _key(x.split("#", 1)[0].strip() or "-") != _key(w)]
    if not lines:
        lines = ["# Ударения пользователя Herald: слово с «+» перед ударной гласной,",
                 "# по одному в строке. Они важнее автоматической расстановки."]
    lines.append(w)
    with open(USER_DICT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return w


def _apply_user(text: str) -> str:
    d = user_stress()
    if not d:
        return text

    def sub(m):
        w = m.group()
        u = d.get(_key(w))
        if u is None:
            return w
        # заглавные буквы - как в тексте, ударение - как у пользователя
        out, j = [], 0
        for ch in u:
            if ch == "+":
                out.append("+")
                continue
            while j < len(w) and w[j] == "+":
                j += 1
            out.append(w[j] if j < len(w) and w[j].lower() == ch else ch)
            j += 1
        return "".join(out)
    return _WORD.sub(sub, text)


def add_stress_text(text: str) -> str:
    """Ударения для подготовки книги: RUAccent + омографы + слова пользователя."""
    return _apply_user(_add_stress_auto(text))


def _add_stress_auto(text: str) -> str:
    acc = _get_accentizer()
    if acc is not None:
        _wrap_classify(acc)
    _CONF.clear()
    out = add_stress(text)
    if out is text or "+" not in out:
        return out
    try:
        # «+» перед «ё» лишний: «ё» всегда ударна. silero-stress его ставит
        # («Е+ё»), RUAccent — нет, и в разметке, на которой дообучали модель,
        # такого сочетания не было; не даём ей незнакомый вход
        return re.sub(r"\+(?=[ёЁ])", "", _homographs_from_silero(out, text))
    except Exception as e:
        log_problem("омографы от silero-stress не применились, оставляю RUAccent", e)
        return out


def unify_book(texts: list[str]) -> tuple[list[str], dict]:
    """Одно ударение на слово вне словаря во всей книге.

    Для каждого слова, которого нет в словаре RUAccent (3,2 млн форм) и которое
    не омограф, собираем варианты ударения по всем частям книги и берём самый
    частый (при равенстве - тот, что встретился раньше). Возвращает тексты и
    сводку {слово: (выбранный вариант, сколько мест исправлено)}.
    """
    from collections import Counter
    acc = _get_accentizer()
    if acc is None:
        return texts, {}
    known = lambda k: k in acc.accents or k in acc.omographs  # noqa: E731
    seen: dict[str, Counter] = {}
    order: dict[tuple, int] = {}
    n = 0
    for t in texts:
        for m in _WORD.finditer(t):
            w = m.group()
            k = _key(w)
            if sum(c in _VOWELS for c in w) < 2 or "ё" in w.lower() or known(k):
                continue
            idx = _stress_idx(w)
            seen.setdefault(k, Counter())[idx] += 1
            order.setdefault((k, idx), n)
            n += 1
    choice = {}
    for k, cnt in seen.items():
        marked = {i: c for i, c in cnt.items() if i}
        if len(cnt) < 2 or not marked:
            continue
        best = max(marked, key=lambda i: (marked[i], -order[(k, i)]))
        choice[k] = best
    if not choice:
        return texts, {}
    fixed: dict[str, list] = {}
    out = []
    for t in texts:
        def sub(m):
            w = m.group()
            k = _key(w)
            if k in choice and _stress_idx(w) != choice[k]:
                fixed.setdefault(k, [None, 0])[1] += 1
                return _apply_idx(w, choice[k])
            return w
        out.append(_WORD.sub(sub, t))
    report = {k: (_apply_idx(k, choice[k]), v[1]) for k, v in fixed.items()}
    return out, report


def to_combining(text_with_plus: str) -> str:
    """'+гласная' -> 'гласная\u0301' (комбинируемое ударение), если движку так удобнее."""
    return re.sub(r"\+([аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "\\1\u0301", text_with_plus)


def available() -> bool:
    return _get_accentizer() is not None
