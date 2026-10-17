"""Проверка слов ПО ЗВУКУ: принудительное выравнивание текста книги (3.3).

Whisper - не «уши», а модель с языковой догадкой: он дописывает правдоподобное
слово по смыслу («Патлся… актив…» -> «активировал»), и сравнение «текст Whisper
против текста книги» ошибки не видит (3.1: 64 из 66 обрубленных слов «услышаны»
целыми). Здесь наоборот: акустическая модель CTC по буквам (wav2vec2 XLSR-53,
русская, без языковой модели) даёт для каждого кадра 20 мс вероятности букв, и
ИЗВЕСТНЫЙ текст книги укладывается на звук (Viterbi, torchaudio.forced_align).
Для каждого слова видно, насколько звук на его месте похож именно на эти
буквы. Съеденное или смазанное слово выравнивание всё равно куда-то положит,
но с низкой вероятностью - это и есть сигнал.

Оценки слова (подбираются по размеченному набору, см. Приёмка 3.3):
  * gop - средняя по кадрам слова «уверенность в своей букве»:
    exp(log P(буква) - log max P(любая)) - 1, если звук и есть эта буква;
  * low - доля букв слова, у которых эта уверенность ниже 0,5;
  * gap_after - пауза после слова (с), для «слитно» на знаке препинания.

Модель: bond005/wav2vec2-large-ru-golos (Apache-2.0; русская XLSR, дообучена на
Golos и Common Voice), локально. Сравнивалась с jonatasgrosman/wav2vec2-large-
xlsr-53-russian (Apache-2.0): на наборе ловят одинаково, golos чуть меньше
срабатывает зря.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

MODEL = os.environ.get("HERALD_ALIGN_MODEL", "bond005/wav2vec2-large-ru-golos")
# Пороги - по размеченному набору 3.3 (места, где Андрей слышал ошибку, и чистые
# места тех же файлов; eval_align_skeleton.py). Слово под подозрением, если его
# буквы почти не слышны (BAD_POST) или модель слышит на его месте другие
# согласные при сниженной вероятности (SOFT_POST и SKELETON_MIN).
BAD_POST = -3.5
SOFT_POST = -0.8
SKELETON_MIN = 0.6
# Растянутая гласная («никогооо», 3.4): голос тянется после выровненного конца
# слова. У голоса Вячеслава на четырёх главах 99-й процентиль - 0,34-0,41 с,
# крайние случаи - 0,54-0,98 с (eval_tail.py). Ловим только крайние: порог
# ниже задевал бы обычные конечные гласные перед запятой.
STRETCH_MAX = 0.6
SR = 16000
FRAME = 0.02                    # wav2vec2: один кадр - 20 мс
WINDOW_SEC = 20.0               # длинный звук - окнами, чтобы не съесть память
OVERLAP_SEC = 2.0


@dataclass
class WordScore:
    word: str                   # как в тексте (без «+»)
    norm: str                   # буквы, которые выравнивались
    start: float                # секунды
    end: float
    gop: float                  # 0..1, средняя уверенность в своих буквах
    low: float                  # доля «неуверенных» букв
    min_char: float             # самая неуверенная буква
    gap_after: float = 0.0      # пауза до следующего слова, с
    punct_after: str = ""       # знак препинания в тексте после слова
    char_gops: list = field(default_factory=list)
    post_mean: float = 0.0      # средняя по буквам наибольшая вероятность буквы у её места
    post_min: float = 0.0       # самая слабая буква
    heard: str = ""             # что модель слышит на месте слова САМА (без языковой модели)
    sim: float = 0.0            # похожесть услышанного на буквы слова, 0..1
    spans: list = field(default_factory=list)   # [(буква, начало с, конец с)] - пики CTC


class Aligner:
    """Загружается один раз; считает на процессоре (видеоядро занято синтезом)."""

    def __init__(self, model: str = MODEL, device: str | None = None):
        self.model_name = model
        self.device = device or "cpu"
        self._m = None

    def load(self):
        if self._m is None:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            # Выравнивание идёт параллельно с синтезом на видеоядре. Если torch
            # займёт все ядра, синтезу нечем отдавать команды видеоядру, и он
            # медленнеет (замер 3.3). Двух потоков выравниванию хватает.
            torch.set_num_threads(int(os.environ.get("HERALD_ALIGN_THREADS", "2")))
            self._proc = Wav2Vec2Processor.from_pretrained(self.model_name)
            self._m = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device).eval()
            vocab = self._proc.tokenizer.get_vocab()
            self._vocab = {k.lower(): v for k, v in vocab.items()}
            self._blank = self._proc.tokenizer.pad_token_id
            self._delim = vocab.get("|")
            self._torch = torch
            # модели учены на тексте без «ё» (Common Voice): «ё» пишется как «е»
            self._ye_as_e = True
        return self

    # --- звук -> логарифмы вероятностей букв по кадрам -----------------
    def emissions(self, audio16: np.ndarray) -> np.ndarray:
        """(кадры, буквы) log-softmax. Длинный звук - окнами с перекрытием."""
        self.load()
        torch = self._torch
        a = np.asarray(audio16, dtype=np.float32)
        win, hop = int(WINDOW_SEC * SR), int((WINDOW_SEC - OVERLAP_SEC) * SR)
        outs = []
        starts = list(range(0, max(1, len(a) - int(OVERLAP_SEC * SR)), hop)) or [0]
        for k, s0 in enumerate(starts):
            piece = a[s0:s0 + win]
            if len(piece) < SR // 10:
                break
            x = self._proc(piece, sampling_rate=SR, return_tensors="pt").input_values
            with torch.inference_mode():
                lp = torch.log_softmax(self._m(x.to(self.device)).logits[0].float(), -1).cpu().numpy()
            if k > 0:                               # отрезать половину перекрытия
                lp = lp[int(OVERLAP_SEC / 2 / FRAME):]
            if k < len(starts) - 1 and s0 + win < len(a):
                lp = lp[:len(lp) - int(OVERLAP_SEC / 2 / FRAME)]
            outs.append(lp)
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(self._vocab)))

    # --- текст -> буквы модели ----------------------------------------
    def _words(self, text: str):
        """[(слово как в тексте, буквы модели, знак препинания после)]."""
        out = []
        t = text.replace("+", "")
        for m in re.finditer(r"[\w\-]+", t):
            for part in m.group(0).split("-"):
                low = part.lower()
                if "ё" not in self._vocab or self._ye_as_e:
                    low = low.replace("ё", "е")
                norm = "".join(c for c in low if c in self._vocab and c != "|")
                if not norm:
                    continue
                tail = t[m.end():m.end() + 3]
                punct = re.match(r"\s*([,.!?…:;—–]|-\s)", tail)
                out.append((part, norm, punct.group(1).strip() if punct else ""))
        return out

    def score(self, audio: np.ndarray, sr: int, text: str) -> list[WordScore]:
        """Оценки слов текста по этому звуку."""
        self.load()
        torch = self._torch
        import torchaudio.functional as AF
        a = _resample(np.asarray(audio, dtype=np.float32).reshape(-1), sr, SR)
        lp = self.emissions(a)
        words = self._words(text)
        if not words or len(lp) == 0:
            return []
        tokens, owner = [], []                      # owner: номер слова у буквы
        for wi, (_w, norm, _p) in enumerate(words):
            if wi and self._delim is not None:
                tokens.append(self._delim); owner.append(-1)
            for c in norm:
                tokens.append(self._vocab[c]); owner.append(wi)
        if len(tokens) > len(lp):                   # звука меньше, чем букв - всё плохо
            return [WordScore(w, n, 0, 0, 0.0, 1.0, 0.0, 0.0, p) for w, n, p in words]
        em = torch.from_numpy(lp)[None]
        tg = torch.tensor([tokens], dtype=torch.int32)
        ali, _sc = AF.forced_align(em, tg, blank=self._blank)
        ali = ali[0].numpy()
        best = lp.max(axis=1)
        # кадры каждой буквы: подряд идущие кадры с её номером
        spans = []                                  # (номер буквы в tokens, начало, конец)
        ti, prev = -1, self._blank
        for f, tok in enumerate(ali):
            if tok == self._blank:
                prev = tok
                continue
            if tok != prev or not spans:
                ti += 1
                spans.append([ti, f, f + 1])
            else:
                spans[-1][2] = f + 1
            prev = tok
        per_word: dict[int, list] = {}
        T = len(lp)
        for ti, f0, f1 in spans:
            if ti >= len(owner) or owner[ti] < 0:
                continue
            tok = tokens[ti]
            conf = float(np.mean(np.exp(lp[f0:f1, tok] - best[f0:f1])))
            # выход CTC «пиковый»: буква - 1-2 кадра среди пауз, поэтому берём её
            # наибольшую вероятность рядом с выровненным местом (±2 кадра)
            post = float(np.exp(lp[max(0, f0 - 2):min(T, f1 + 2), tok].max()))
            per_word.setdefault(owner[ti], []).append((f0, f1, conf, post, ti))
        inv = {v: k for k, v in self._vocab.items()}
        out = []
        for wi, (w, norm, p) in enumerate(words):
            ch = per_word.get(wi, [])
            if not ch:
                out.append(WordScore(w, norm, 0, 0, 0.0, 1.0, 0.0, 0.0, p))
                continue
            confs = [c[2] for c in ch]
            posts = [c[3] for c in ch]
            f0, f1 = ch[0][0], ch[-1][1]
            # что модель слышит сама: жадное декодирование кадров слова (±1 кадр)
            g = lp[max(0, f0 - 1):min(T, f1 + 1)].argmax(axis=1)
            heard, prev = [], None
            for tok in g:
                if tok != prev and tok != self._blank and tok != self._delim:
                    heard.append(inv.get(int(tok), ""))
                prev = tok
            heard = "".join(heard)
            import difflib
            sim = difflib.SequenceMatcher(None, norm, heard).ratio()
            out.append(WordScore(w, norm, f0 * FRAME, f1 * FRAME,
                                 float(np.mean(confs)), float(np.mean([c < 0.5 for c in confs])),
                                 float(min(confs)), 0.0, p, confs,
                                 float(np.mean(np.log(np.maximum(posts, 1e-6)))), float(min(posts)),
                                 heard, float(sim),
                                 [(inv.get(tokens[c[4]], ""), c[0] * FRAME, c[1] * FRAME) for c in ch]))
        for i in range(len(out) - 1):
            out[i].gap_after = max(0.0, out[i + 1].start - out[i].end)
        return out


_VOW = set("аеёиоуыэюяйьъ")


def skeleton(x: str) -> str:
    """Согласный скелет: гласные, й, ь, ъ выброшены - безударные гласные русская
    речь редуцирует («абориген» -> «обреген»), это не ошибка произношения."""
    return "".join(c for c in x.lower().replace("ё", "е") if c.isalpha() and c not in _VOW)


def skeleton_sim(word: str, heard: str) -> float:
    """Похожесть согласных слова книги и услышанного моделью, 0..1."""
    import difflib
    a, b = skeleton(word), skeleton(heard)
    if not a:
        return 1.0 if not b or len(b) <= 1 else 0.5
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_bad(ws: WordScore) -> bool:
    """Слово съедено, смазано или заменено - по звуку (пороги выше).

    Однобуквенные слова («и», «в», «о», «а», «с») не судим (3.7): в беглой речи
    они сливаются с соседом, и выравнивание по буквам их «не слышит». На «Пробе
    пера» они давали треть пересчётов (каждый ~10 с счёта), а в размеченном
    наборе 3.3 ни одной настоящей ошибки на однобуквенном слове не было."""
    if len(ws.norm) <= 1:
        return False
    return ws.post_mean < BAD_POST or (ws.post_mean < SOFT_POST
                                       and skeleton_sim(ws.word, ws.heard) < SKELETON_MIN)


def stretched(ws: list[WordScore], audio: np.ndarray, sr: int,
              limit: float = STRETCH_MAX, quiet_db: float = -35.0) -> list[str]:
    """Слова, после которых голос звучит дольше limit (растянутая гласная).

    Звучание - громкие кадры по 10 мс между концом слова (пик последней буквы)
    и началом следующего; у последнего слова куска - до конца звука.
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    w = max(1, int(0.01 * sr))
    n = len(a) // w
    if n < 10 or not ws:
        return []
    e = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1) + 1e-12)
    loud = 20 * np.log10(e / np.percentile(e, 95)) > quiet_db
    out = []
    for i, x in enumerate(ws):
        if not x.norm or x.end <= 0:
            continue
        t1 = ws[i + 1].start if i + 1 < len(ws) else n * 0.01
        k0, k1 = int(x.end / 0.01), int(t1 / 0.01)
        if k1 > k0 and loud[k0:k1].sum() * 0.01 > limit:
            out.append(x.word)
    return out


LAST_WORD_MIN = 0.6
_DEVOICE = str.maketrans("бвгджз", "пфктшс")


def _end_skeleton(word: str) -> str:
    """Скелет с оглушённой последней согласной, если слово ею кончается:
    звонкая на конце оглушается и у живого диктора («интриг» - [интрик])."""
    w = word.lower().rstrip("ьъ")
    sk = skeleton(w)
    if sk and w and w[-1] not in _VOW:
        sk = sk[:-1] + sk[-1].translate(_DEVOICE)
    return sk


def edges_ok(ws: list[WordScore], text: str) -> bool:
    """Края куска по звуку (3.5) - вместо распознавания краёв Whisper.

    Первое слово не должно быть съедено (is_bad). У последнего слова должен
    прозвучать КОНЕЦ: последняя согласная (или две) его скелета - те же, что
    слышит акустическая модель на его месте. Так ловится «ждать» -> «жда»,
    «интриг» -> «интри»; безударные гласные на конце не сравниваются (их
    редуцирует и живой диктор). Многоточие в конце - автор сам оборвал слово
    («Не выйде…»): тогда только «не съедено». Слова короче 4 букв - как у
    прежней проверки - не судим: на них модель путается сама.
    """
    words = [w for w in ws if w.norm]
    if not words:
        return True
    first, last = words[0], words[-1]
    if len(first.norm) >= 4 and is_bad(first):
        return False
    if len(last.norm) >= 4:
        if is_bad(last):
            return False
        if not text.rstrip().endswith(("...", "…")):
            # слово заменено похожим по согласным («выйдет» -> «вредит», поймано
            # test_endings): по всем буквам. Порог 0,6 - ниже него на живой речи
            # (742 чистых слова набора 3.3) не было ни одного верного слова
            import difflib
            if difflib.SequenceMatcher(None, last.norm, last.heard).ratio() < LAST_WORD_MIN:
                return False
            want, got = _end_skeleton(last.norm), _end_skeleton(last.heard)
            k = 2 if len(want) >= 4 else 1
            if want and got[-k:] != want[-k:]:
                return False
    return True


def _resample(a: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return a
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr, to)
    return resample_poly(a, to // g, sr // g).astype(np.float32)


_ALIGNER: Aligner | None = None


def aligner() -> Aligner:
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = Aligner()
    return _ALIGNER
