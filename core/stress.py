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


_TITLE = re.compile(r"\b([А-ЯЁ])([а-яё]+)\b")


def _lower_titles(text: str) -> str:
    """«Северная» -> «северная»: слова с заглавной RUAccent размечает хуже (карточка
    ru-stress-cf: 95,2 % против 97,4 % в нижнем регистре; 3.8, задание 10а).
    АББРЕВИАТУРЫ не трогаем."""
    return _TITLE.sub(lambda m: m.group(1).lower() + m.group(2), text)


def _restore_case(src: str, out: str) -> str:
    """Регистр букв - как в исходном тексте; «+» из разметки сохраняются. Если
    буквы разошлись (RUAccent что-то поменял) - разметка как есть."""
    res, j = [], 0
    for ch in out:
        if ch == "+":
            res.append(ch)
            continue
        if j < len(src) and src[j].lower() == ch.lower():
            res.append(src[j])
            j += 1
        else:
            return out
    return "".join(res) if j == len(src) else out


# 11а, п. 1: размечать по одному предложению. Раньше RUAccent получал главу целиком
# после _lower_titles, а предложения он находит по заглавной букве после точки - и
# 55 предложений главы склеивались в 8 «предложений» до 2 500 знаков: модели
# падали на 512 токенах, омограф решался по обрезанному тексту («к Мал+ому
# Доминиону», хотя в одном предложении RUAccent уверенно ставит «м+алому»).
BY_SENTENCE = True
# Соседние предложения - контекст для выбора омографа (знаки - только своему).
# Выключено замером 11а: на 50 и 300 словах ничего не дало, а в длинном куске
# 3.9 предыдущая фраза сбила «Св+язи требует» на «Связ+и».
NEIGHBORS = False
CTX_CHARS = 300
_CTX = {"before": "", "after": ""}


def sentences(text: str) -> list[tuple[int, int]]:
    """Границы предложений (по исходному регистру - так их видит razdel)."""
    try:
        from razdel import sentenize
        return [(s.start, s.stop) for s in sentenize(text)]
    except Exception:
        return [(m.start(), m.end()) for m in re.finditer(r"[^.!?…\n]+[.!?…]*", text)]


def add_stress(text: str) -> str:
    """Вернуть текст с ударениями ('+' перед ударной гласной). При сбое - как есть."""
    if not text or not text.strip():
        return text
    acc = _get_accentizer()
    if acc is None:
        return text
    if BY_SENTENCE:
        return _fix_dialog_dash(_by_sentence(text, lambda s, low, b, a: _ruaccent_one(acc, s, low)))
    try:
        return _fix_dialog_dash(_restore_case(text, acc.process_all(_lower_titles(text))))
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


def _by_sentence(text: str, one) -> str:
    """one(предложение, оно же строчными, до, после) -> размеченное предложение;
    промежутки между предложениями (пробелы, переносы абзацев) - как были."""
    spans = sentences(text)
    low = [_lower_titles(text[a:b]) for a, b in spans]
    out, pos = [], 0
    for i, (a, b) in enumerate(spans):
        out.append(text[pos:a])
        before = low[i - 1][-CTX_CHARS:] if i and NEIGHBORS else ""
        after = low[i + 1][:CTX_CHARS] if i + 1 < len(spans) and NEIGHBORS else ""
        _CTX["before"], _CTX["after"] = before, after
        try:
            out.append(one(text[a:b], low[i], before, after))
        finally:
            _CTX["before"] = _CTX["after"] = ""
        pos = b
    out.append(text[pos:])
    return "".join(out)


def _ruaccent_one(acc, src: str, low: str) -> str:
    """Одно предложение через RUAccent. Сверхдлинное (больше 512 токенов) - по
    частям между запятыми; не вышло - без ударений, с записью в журнал."""
    try:
        return _restore_case(src, acc.process_all(low))
    except Exception as e:
        parts = re.split(r"(?<=[,;:])\s+", src)
        if len(parts) > 1:
            try:
                return " ".join(_restore_case(p, acc.process_all(_lower_titles(p))) for p in parts)
            except Exception:
                pass
        log_problem(f"фраза осталась без ударений: «{src[:60]}»", e)
        return src


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
SILERO_BELOW = 0.0
# 11а: подмена выключена. С разметкой по предложению (и строчными буквами, 3.8)
# RUAccent сам вернее: на 90 омографах 85 верно, с подменой silero - 81 (silero
# помог в 2 местах, испортил 6: «стены», «печи», «проводах», «Связи требует»…).
# 3.7 подбирал 0,8 ещё на главе одним куском.
# Слова, где RUAccent и silero-stress поставили разные ударения (3.8, задание 10а,
# п. 3.1): «Как будет прочитано» их подсвечивает - только просмотр, правки нет.
DISPUTED: set[str] = set()


def take_disputed() -> list[str]:
    """Спорные слова, накопленные с прошлого вызова (и очистить список)."""
    out = sorted(DISPUTED)
    DISPUTED.clear()
    return out


def _wrap_classify(acc) -> None:
    om = getattr(acc, "omograph_model", None)
    if om is None or getattr(om, "_herald_conf", False):
        return
    import numpy as np

    def classify(texts, hypotheses, num_hypotheses):
        outs, pos = [], 0
        for n in num_hypotheses:
            t = re.sub(r"\s+(?=(?:[,.?!:;…]))", "", texts[pos])
            if _CTX["before"] or _CTX["after"]:    # 11а: соседние предложения - контекст
                t = " ".join(x for x in (_CTX["before"], t, _CTX["after"]) if x)
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
            if "+" in m.group() and "+" in w and _stress_idx(m.group()) != _stress_idx(w):
                DISPUTED.add(k)
            if ((k in homos or k in yo_homos) and _stress_idx(m.group()) != _stress_idx(w)
                    and _CONF.get(k, 0.0) < SILERO_BELOW):
                repl.append((m.start(), m.end(), w))
    for a, b, w in reversed(repl):
        ru_out = ru_out[:a] + w + ru_out[b:]
    return ru_out


def add_stress_text(text: str) -> str:
    """Ударения для подготовки книги: RUAccent + омографы (3.8: ручного словаря нет -
    решение Андрея 17.10, всё автоматически)."""
    return _add_stress_auto(text)


# --- 11а, пп. 2-3: спорное - автоматически, по контексту ---------------------
# Омограф (и «все/всё»), где RUAccent не уверен или разошёлся с silero-stress,
# решает судья по предложению с соседними (core/homograph_llm, локальная модель
# Ollama), если он есть; иначе - как в 3.7 (silero, когда RUAccent не уверен).
JUDGE = False                   # замер 11а: модели решают омографы хуже RUAccent (77 против 85 из 90)
JUDGE_BELOW = 0.9               # уверенность RUAccent, ниже которой омограф - сомнительный
# Обычное слово, где RUAccent и silero-stress разошлись («щавель», «красивее»), -
# по справочнику ударений (core/stressdict, Викисловарь); если справочник знает
# несколько ударений формы - судья, иначе RUAccent.
REFERENCE = True
# И без спора: обычное слово (не омограф), у которого словарь Зализняка знает одно
# ударение, а RUAccent поставил другое («что-ниб+удь» -> «чт+о-нибудь»).
REFERENCE_ALL = True
# «Омограф» RUAccent с одним ударением по Викисловарю - по словарю. Выключено:
# на наборах 11а ничего не исправило, а «т+янущуюся» испортило (шум словаря).
SINGLE_FORM = False
# Омограф, у которого формы различаются числом (сте́ны им. мн. - стены́ род. ед.,
# во́ды - воды́, ру́ки - руки́): прилагательное или причастие прямо перед ним
# однозначно во множественном числе - форма множественного, в единственном - родительный
# единственного. Морфология - pymorphy3 (MIT, уже в программе через ru-normalizr).
AGREEMENT = True
_morph: list = []
# Решения судьи: (слово, предложение) -> вариант. Конвейер кладёт их в служебную
# папку книги (п. 4): повторная разметка той же книги не зовёт модель и даёт то же.
JUDGED: dict[str, str] = {}
# п. 4: что решено по книге - форма -> {ударение, источник, мест}; конвейер сохраняет
# в _служебное/ударения книги.json (take_decisions)
DECISIONS: dict[str, dict] = {}
JUDGE_STATS = {"сомнительных": 0, "судья": 0, "справочник": 0}
_judge_model: list = []


def _judge():
    """Модель-судья или None (Ollama не запущена / модели нет)."""
    if not JUDGE:
        return None
    if not _judge_model:
        try:
            from .homograph_llm import available_model
            _judge_model.append(available_model())
        except Exception:
            _judge_model.append(None)
    return _judge_model[0]


def _token_variant(token: str, variant: str) -> str:
    """Слово текста в виде варианта: регистр - как в тексте, «ё» и «+» - как в варианте."""
    plain = [c for c in variant if c != "+"]
    letters = [c for c in token if c != "+"]
    if len(plain) != len(letters):
        return token
    out = []
    for c, v in zip(letters, plain):
        if v in "ёЁ" and c in "еЕ":
            c = "Ё" if c.isupper() else "ё"
        elif v in "еЕ" and c in "ёЁ":
            c = "Е" if c.isupper() else "е"
        out.append(c)
    return _apply_idx("".join(out), _stress_idx(variant) if "+" in variant else ())


def _variants(acc, k: str) -> tuple[list[str], list[set]]:
    """Варианты слова с грамматикой: справочник; нет - варианты омографа RUAccent."""
    vs, gr = [], []
    if REFERENCE:
        try:
            from . import stressdict
            v = stressdict.variants(k)
            if len({stressdict.stress_of(x) for x in v}) > 1 or any("ё" in x for x in v) and k in v:
                vs = sorted(v)
                gr = [v[x] for x in vs]
        except Exception:
            vs = []
    if len(vs) < 2:
        vs = list(acc.omographs.get(k, []))
        yo = getattr(acc, "yo_homographs", {}).get(k)
        if yo:
            vs = [k, yo]
        gr = [set() for _ in vs]
    return vs, gr


def _resolve(acc, ru: str, src: str, before: str, after: str) -> str:
    """Предложение после RUAccent -> спорные слова решены (пп. 2-3)."""
    import difflib
    homos = acc.omographs
    yo_homos = getattr(acc, "yo_homographs", {})
    si = _silero_text(src) if _get_silero() is not None else ""
    rm = list(_WORD.finditer(ru))
    sw = _WORD.findall(si)
    sm = difflib.SequenceMatcher(None, [_key(m.group()) for m in rm], [_key(w) for w in sw], autojunk=False)
    pair = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for m, w in zip(rm[i1:i2], sw[j1:j2]):
                pair[m.start()] = w
    repl = []
    for m in rm:
        tok, k = m.group(), _key(m.group())
        w = pair.get(m.start())
        is_yo = k in yo_homos
        if is_yo:
            differ = w is not None and (("ё" in tok.lower()) != ("ё" in w.lower()))
        else:
            differ = (w is not None and "+" in tok and "+" in w and _stress_idx(tok) != _stress_idx(w))
        homo = k in homos or is_yo
        if differ and not is_yo:
            DISPUTED.add(k)
        conf = _CONF.get(k, 0.0)        # нет оценки (омограф на «ё») - как в 3.7: не уверен
        new = None
        single = _single_form(k) if (SINGLE_FORM and homo and not is_yo and "+" in tok) else None
        if single is not None:
            # «омограф» RUAccent, у которого по словарям одно ударение («м+алому» -
            # «мал+ому» у RUAccent): решает словарь
            if tuple(sorted(_stress_idx(tok))) != single:
                new = _apply_idx(tok, single)
                JUDGE_STATS["справочник"] += 1
                _decide(k, new, "словарь (у формы одно ударение)")
        elif (agree := _by_agreement(k, ru, m.start()) if (AGREEMENT and homo and not is_yo
                                                             and "+" in tok) else None) is not None:
            # согласование с прилагательным перед словом («Холодные в+оды», «этой стен+ы»)
            if tuple(sorted(_stress_idx(tok))) != agree:
                new = _apply_idx(tok, agree)
                _decide(k, new, "согласование с прилагательным")
        elif homo and (differ or (k in homos and conf < JUDGE_BELOW)):
            JUDGE_STATS["сомнительных"] += 1
            got = _ask_judge(acc, tok, k, src, before, after)
            if got is not None:
                new = got
            elif differ and conf < SILERO_BELOW:
                new = w                                  # как в 3.7
        elif differ and REFERENCE:
            new = _from_reference(acc, tok, k, src, before, after)
        elif REFERENCE_ALL and not homo and "+" in tok:
            new = _from_zaliznyak(tok, k)
        if new is not None and new != tok:
            repl.append((m.start(), m.end(), new))
    for a, b, w in reversed(repl):
        ru = ru[:a] + w + ru[b:]
    return ru


def _ask_judge(acc, tok: str, k: str, src: str, before: str, after: str) -> str | None:
    model = _judge()
    if model is None:
        return None
    vs, gr = _variants(acc, k)
    if len(vs) < 2:
        return None
    ck = f"{k}\t{src.strip()}"
    if ck not in JUDGED:
        from .homograph_llm import choose_in_context
        n = choose_in_context(tok.replace("+", ""), src, vs, before, after, gr, model=model)
        if n is None:
            return None
        JUDGED[ck] = vs[n]
        JUDGE_STATS["судья"] += 1
    return _token_variant(tok, JUDGED[ck])


def _from_reference(acc, tok: str, k: str, src: str, before: str, after: str) -> str | None:
    try:
        from . import stressdict
        u = stressdict.unique(k)
    except Exception:
        return None
    if u:
        JUDGE_STATS["справочник"] += 1
        new = _apply_idx(tok, u)
        _decide(k, new, "словарь (RUAccent и silero разошлись)")
        return new
    return _ask_judge(acc, tok, k, src, before, after) if len(_variants(acc, k)[0]) > 1 else None


def _decide(k: str, new: str, how: str) -> None:
    d = DECISIONS.setdefault(k, {"ударение": new.lower(), "источник": how, "мест": 0})
    d["мест"] += 1


def take_decisions() -> dict:
    out = dict(DECISIONS)
    DECISIONS.clear()
    return out


def _by_agreement(k: str, text: str, pos: int) -> tuple | None:
    """Ударение омографа по числу прилагательного перед ним или None."""
    try:
        from . import stressdict
        opts = stressdict.options(k)
        if len(opts) != 2:
            return None
        plur = [o for o, t in opts.items() if "мн" in t and ("им" in t or "вин" in t)]
        sing = [o for o, t in opts.items() if "мн" not in t and "ед" in t and "род" in t]
        if len(plur) != 1 or len(sing) != 1:
            return None
        prev = re.findall(r"([А-Яа-яЁё+-]+)\s*$", text[:pos])
        if not prev or re.search(r"[,.;:!?…—–-]\s*$", text[:pos]):
            return None
        if not _morph:
            import pymorphy3
            _morph.append(pymorphy3.MorphAnalyzer())
        parses = _morph[0].parse(prev[0].replace("+", "").lower())
        adj = [p for p in parses if p.tag.POS in ("ADJF", "PRTF")]
        if not adj or sum(p.score for p in adj) < 0.2:
            return None
        nums = {p.tag.number for p in parses if p.tag.POS in ("ADJF", "PRTF", "NOUN") and p.tag.number}
        if nums == {"plur"}:
            return plur[0]
        if nums == {"sing"}:
            return sing[0]
    except Exception:
        return None
    return None


def _single_form(k: str) -> tuple | None:
    """Одно ударение формы по Викисловарю, и Зализняк не спорит, - иначе None.
    Одного Зализняка для омографов мало: в выборке у «атлас», «орган», «ирис»,
    «отмели» только одно из двух ударений."""
    try:
        from . import stressdict
        o = stressdict.options(k)
        if len(o) != 1:
            return None
        u = next(iter(o))
        z = stressdict.zaliznyak(k)
        return u if (z is None or z == u) else None
    except Exception:
        return None


def _from_zaliznyak(tok: str, k: str) -> str | None:
    try:
        from . import stressdict
        z = stressdict.zaliznyak(k)
    except Exception:
        return None
    if z and tuple(sorted(_stress_idx(tok))) != z and not any(c in "ёЁ" for c in tok):
        JUDGE_STATS["справочник"] += 1
        new = _apply_idx(tok, z)
        _decide(k, new, "словарь Зализняка (RUAccent ошибся)")
        return new
    return None


def _one_sentence(acc, src: str, low: str, before: str, after: str) -> str:
    _CONF.clear()
    ru = _ruaccent_one(acc, src, low)
    if "+" not in ru:
        return ru
    try:
        out = _resolve(acc, ru, src, before, after)
    except Exception as e:
        log_problem("спорные ударения не решились, оставляю RUAccent", e, show=False)
        out = ru
    # «+» перед «ё» лишний (см. ниже)
    return re.sub(r"\+(?=[ёЁ])", "", out)


def _add_stress_auto(text: str) -> str:
    acc = _get_accentizer()
    if acc is not None:
        _wrap_classify(acc)
    if BY_SENTENCE and acc is not None and text and text.strip():
        return _fix_dialog_dash(_by_sentence(text, lambda s, low, b, a: _one_sentence(acc, s, low, b, a)))
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
