"""Звук сразу, по мере готовности (3.4): синтез по кускам и игра из очереди.

Один механизм для окна (пробник, «Прослушать», «Прочитать вслух»), консоли
(`--speak`) и локального сервиса `/speak` (core/speak_server.py).

Как устроено:
  * текст режется на куски как в книге (`_chunk_paragraphs`), только первый
    кусок короткий - одно предложение до FIRST_MAX знаков, чтобы первый звук
    пришёл быстро;
  * куски считает обычный `_synth_chunks` (те же ударения, проверки, паузы,
    кэш) и отдаёт их сюда по одному, уже склеенными тем же кодом, что и файл:
    звук потока совпадает с итоговым файлом до сэмпла;
  * `Live` решает, когда начинать играть. Клон F5 считает медленнее реального
    времени, и если начать сразу, звук будет запинаться. Старт - когда
    оставшийся счёт успеет до конца звучания (с запасом MARGIN), по скорости,
    замеренной на этом движке и голосе (`Speed`, запоминается между запусками);
  * если машина всё же не успела, пауза выйдет на стыке кусков (кусок играется,
    только когда посчитан целиком) - не посреди слова;
  * полировка (assemble.HD_FILTER) и поправки голоса (voicefx) - ffmpeg на
    каждый кусок отдельно. Кусок начинается и кончается тишиной, поэтому
    фильтрам не на чем «щёлкнуть» на стыке (замер - в Приёмке 3.4).
"""

from __future__ import annotations

import collections
import json
import os
import subprocess
import threading
import time

import numpy as np

OUT_SR = 48000                 # частота на выходе полировки (как у книги)
MARGIN = 0.10                  # запас к расчёту старта
FIRST_MAX = 100                # первый кусок - до стольких знаков…
FIRST_MIN = 40                 # …но не огрызок: на коротком куске F5 мажет первое слово
CHARS_PER_SEC = 12.0           # оценка длины звука до первого замера (chapters)
SPEED_FILE = os.path.join(os.path.expanduser("~"), ".cache", "text2audio", "live_speed.json")
KEEP_OBS = 40

# Движок не потокобезопасен: одновременно считает только один заказ (окно,
# сервис, консоль - в одном процессе). Остальные ждут или получают «занято».
ENGINE_LOCK = threading.RLock()


# --- скорость движка с этим голосом -------------------------------------
class Speed:
    """Время счёта куска = a + b × длина его звука; длина звука = знаки × spc.

    a - постоянная часть: F5 на каждый вызов заново прогоняет образец голоса,
    поэтому короткая фраза стоит дороже, чем по RTF длинной главы. Замеры
    копятся по ключу «движок + голос + проверка» в SPEED_FILE.
    """

    def __init__(self, key: str, prior_rtf: float, path: str | None = SPEED_FILE,
                 prior_fixed: float = 0.0):
        """path=None - замеры не читаются и не сохраняются (проверки)."""
        self.key, self.prior, self.path = key, float(prior_rtf), path
        self.prior_fixed = float(prior_fixed)
        self.obs: list[tuple[float, float]] = []
        self.spc = 1.0 / CHARS_PER_SEC
        # доля «настоящее время счёта / сумма стоимостей кусков»: с упреждением
        # проверка одного куска идёт одновременно со счётом следующего
        self.overlap = 1.0
        try:
            if path is None:
                raise OSError
            d = json.load(open(path, encoding="utf-8")).get(key, {})
            self.obs = [tuple(x) for x in d.get("obs", [])][-KEEP_OBS:]
            self.spc = float(d.get("spc", self.spc))
            self.overlap = float(d.get("overlap", 1.0))
        except (OSError, ValueError, TypeError):
            pass                        # нет замеров - берём прикидку профиля

    def _fit(self) -> tuple[float, float]:
        if not self.obs:
            return self.prior_fixed, self.prior
        pts = list(self.obs)
        if len(pts) < 6:                # мало замеров - прикидка профиля в помощь
            pts += [(d, self.prior_fixed + self.prior * d) for d in (3.0, 12.0)]
        x = np.array([o[0] for o in pts])
        y = np.array([o[1] for o in pts])
        if len(x) >= 3 and x.std() > 0.5:
            b, a = np.polyfit(x, y, 1)
            if a >= 0 and b > 0:
                return float(a), float(b)
        return 0.0, float(y.sum() / max(1e-6, x.sum()))

    def synth_sec(self, audio_sec: float, overlapped: bool = True) -> float:
        a, b = self._fit()
        return (a + b * audio_sec) * (self.overlap if overlapped else 1.0)

    def learn_overlap(self, wall: float, cost_sum: float) -> None:
        if wall > 0 and cost_sum > 0:
            ov = float(np.clip(wall / cost_sum, 0.4, 1.0))
            # осторожно: берём большее из старого и нового - лучше начать позже
            self.overlap = ov if self.overlap >= 1.0 else max(ov, 0.5 * (self.overlap + ov))

    def audio_sec(self, chars: int, tempo: float = 1.0) -> float:
        """Длина звука; spc хранится для темпа 1,0, медленнее - звука больше."""
        return chars * self.spc / max(0.3, tempo)

    @property
    def rtf(self) -> float:
        x = sum(o[0] for o in self.obs)
        return sum(o[1] for o in self.obs) / x if x else self.prior

    def observe(self, chars: int, audio_sec: float, synth_sec: float,
                tempo: float = 1.0) -> None:
        if audio_sec <= 0.2:
            return
        self.obs = (self.obs + [(round(audio_sec, 3), round(synth_sec, 3))])[-KEEP_OBS:]
        if chars > 0:
            self.spc = 0.7 * self.spc + 0.3 * (audio_sec * tempo / chars)

    def save(self) -> None:
        # только обычный файл: os.replace поверх устройства (/dev/null под root
        # в контейнере) подменил бы его - так было на первой проверке в Docker
        if self.path is None or (os.path.exists(self.path) and not os.path.isfile(self.path)):
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            try:
                allv = json.load(open(self.path, encoding="utf-8"))
            except (OSError, ValueError):
                allv = {}
            allv[self.key] = {"obs": self.obs, "spc": round(self.spc, 5),
                              "overlap": round(self.overlap, 3)}
            tmp = self.path + ".tmp"
            json.dump(allv, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass                        # не записали - в следующий раз прикидка


def wait_needed(ready: float, rest: list[tuple[float, float]], elapsed: float = 0.0,
                margin: float = MARGIN) -> float:
    """Сколько ещё ждать до старта, с.

    ready - звук, уже готовый к игре; rest - [(время счёта, длина звука)]
    оставшихся кусков по порядку; elapsed - сколько уже идёт счёт текущего.
    Кусок j должен досчитаться раньше, чем до него дойдёт игра: если начать
    после k следующих кусков, условие для каждого j > k -
        счёт(k+1..j) × (1 + запас) ≤ готово + звук(k+1..j-1).
    """
    m = 1.0 + margin
    if rest and elapsed > 0:            # текущий кусок уже частично посчитан
        rest = [(max(0.0, rest[0][0] - elapsed / m), rest[0][1])] + list(rest[1:])
    for k in range(len(rest) + 1):
        have = ready + sum(d for _s, d in rest[:k])
        ok, synth, play = True, 0.0, have
        for s, d in rest[k:]:
            synth += s * m
            if synth > play:
                ok = False
                break
            play += d
        if ok:
            return sum(s for s, _d in rest[:k]) * m
    return sum(s for s, _d in rest) * m


# --- куда играть ----------------------------------------------------------
class _Sink:
    """Общее для выходов: очередь кусков, пауза, учёт запинок."""

    def __init__(self):
        self.sr = OUT_SR
        self._q: collections.deque = collections.deque()
        self._cur, self._pos = None, 0
        self._lock = threading.Lock()
        self.finished = False           # больше кусков не будет
        self.paused = False
        self.stopped = False
        self.done = threading.Event()
        self.played = 0                 # сэмплов отдано
        self.stall_sec = 0.0
        self.stalls = 0
        self._in_stall = False
        self.chunks_out: list[np.ndarray] = []

    def open(self, sr: int) -> None:
        self.sr = sr

    def write(self, a: np.ndarray) -> None:
        with self._lock:
            self._q.append(np.asarray(a, dtype=np.float32))

    def finish(self) -> None:
        self.finished = True

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.stopped = True
        with self._lock:
            self._q.clear()
            self._cur = None
        self.done.set()

    def _pull(self, n: int) -> tuple[np.ndarray, bool]:
        """n сэмплов для вывода и признак «всё сыграно»."""
        out = np.zeros(n, dtype=np.float32)
        if self.paused or self.stopped:
            return out, self.stopped
        got = 0
        with self._lock:
            while got < n:
                if self._cur is None:
                    if not self._q:
                        break
                    self._cur, self._pos = self._q.popleft(), 0
                    self.chunks_out.append(self._cur)
                take = min(n - got, len(self._cur) - self._pos)
                out[got:got + take] = self._cur[self._pos:self._pos + take]
                got += take
                self._pos += take
                if self._pos >= len(self._cur):
                    self._cur = None
            empty = self._cur is None and not self._q
        self.played += got
        if got < n and not (self.finished and empty):
            self.stall_sec += (n - got) / self.sr     # ждём кусок: тишина на стыке
            if not self._in_stall:
                self.stalls += 1
            self._in_stall = True
        elif got:
            self._in_stall = False
        return out, self.finished and empty and got < n

    @property
    def position(self) -> float:
        return self.played / self.sr


class SoundSink(_Sink):
    """Звуковая карта через sounddevice (PortAudio) - Mac, Windows, Linux."""

    BLOCK = 1024

    def __init__(self):
        super().__init__()
        import sounddevice  # noqa: F401 - нет пакета/устройства - пусть упадёт здесь
        self._stream = None

    def open(self, sr: int) -> None:
        import sounddevice as sd
        super().open(sr)

        def cb(outdata, frames, _t, _status):
            a, end = self._pull(frames)
            outdata[:, 0] = a
            if end:
                raise sd.CallbackStop
        self._stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32",
                                       blocksize=self.BLOCK, callback=cb,
                                       finished_callback=self.done.set)
        self._stream.start()

    def stop(self) -> None:
        super().stop()
        s, self._stream = self._stream, None
        if s is not None:
            try:
                s.abort()
                s.close()
            except Exception:
                pass                    # поток уже закрыт


class ClockSink(_Sink):
    """Без звуковой карты: «играет» по часам. Для проверок, CI и замеров."""

    def __init__(self, realtime: bool = True):
        super().__init__()
        self.realtime = realtime

    def open(self, sr: int) -> None:
        super().open(sr)

        def run():
            block = 1024
            t0 = time.monotonic()
            while not self.stopped:
                _a, end = self._pull(block)
                if end:
                    break
                if self.realtime:
                    t0 += block / sr
                    time.sleep(max(0.0, t0 - time.monotonic()))
            self.done.set()
        threading.Thread(target=run, daemon=True, name="clock-sink").start()


class CallbackSink(_Sink):
    """Отдаёт куски сразу, как пришли (сервис: звук потоком в другую программу)."""

    def __init__(self, send):
        super().__init__()
        self._send = send

    def write(self, a: np.ndarray) -> None:
        if self.stopped:
            return
        a = np.asarray(a, dtype=np.float32)
        self.chunks_out.append(a)
        self.played += len(a)
        try:
            self._send(a)
        except Exception:
            self.stop()                 # клиент ушёл - останавливаем и синтез

    def finish(self) -> None:
        super().finish()
        self.done.set()


def default_sink() -> _Sink | None:
    """Звуковая карта; None - если вывести звук потоком нельзя."""
    try:
        return SoundSink()
    except Exception:
        return None


# --- управление: когда играть --------------------------------------------
class Live:
    """Мост между синтезом и выходом. Синтез зовёт plan/feed/finish."""

    def __init__(self, sink: _Sink | None, speed: Speed | None = None, *,
                 hd: bool = True, fx_chain: str | None = None, on_status=None,
                 margin: float = MARGIN, gate: bool = True):
        self.sink = sink
        self.speed = speed or Speed("-", 1.0, path=None)
        self.hd, self.fx_chain = hd, fx_chain
        self.on_status = on_status
        self.margin, self.gate = margin, gate
        self.t0 = time.monotonic()
        self.sr = None
        self.plan_items: list[tuple[int, float]] = []   # (знаков, пауза) кусков
        self.fed = 0                     # сколько кусков пришло
        self.ready_sec = 0.0             # звука готово
        self.raw: list[np.ndarray] = []
        self.started = False
        self.first_sound: float | None = None
        self.finished = False
        self.stopped = False
        self.paused = False
        self._t_last = None
        self.tempo = 1.0                 # темп синтеза (клон): меняется на ходу
        self._out_lens: list[int] = []   # длины кусков на выходе - для «что звучит»
        self._pending: list[np.ndarray] = []     # готово, но старт ещё не наступил
        self.timeline: list[tuple] = []
        self._start_lock = threading.Lock()
        self._out: collections.deque = collections.deque()
        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._run, daemon=True, name="live-polish")
        self._worker.start()
        self._ticker = threading.Thread(target=self._tick, daemon=True, name="live-tick")
        self._ticker.start()
        self.error: Exception | None = None

    # --- хуки синтеза ---
    def plan(self, items: list[tuple[str, float]], sr: int) -> None:
        self.sr = sr
        self._t_plan = time.monotonic()
        self._cost_sum = 0.0
        self.plan_items = [(_letters(t), g) for t, g in items]
        self._t_last = time.monotonic()
        self._status()

    def feed(self, seg: np.ndarray, chunk_sec: float, cost: float | None = None) -> None:
        """cost - сколько стоил кусок (счёт и проверки); без него - интервал."""
        now = time.monotonic()
        k = self.fed
        if k < len(self.plan_items) and self._t_last is not None and chunk_sec > 0:
            c = cost if cost is not None else now - self._t_last
            self.speed.observe(self.plan_items[k][0], chunk_sec, c, self.tempo)
            self._cost_sum += c
        self._t_last = now
        self.fed += 1
        self.raw.append(seg)
        # ход прогона (для замеров): когда кусок вышел, сколько звука, что предсказывали
        self.timeline.append((round(now - self.t0, 2), round(len(seg) / float(self.sr or 1), 2),
                              round(self.ready_sec, 2)))
        with self._cv:
            self._out.append(seg)
            self._cv.notify_all()

    def finish(self) -> None:
        with self._cv:
            self.finished = True
            self._cv.notify_all()
        if self.plan_items and len(self.plan_items) > 1:
            self.speed.learn_overlap(time.monotonic() - self._t_plan, self._cost_sum)
        self.speed.save()

    def should_stop(self) -> bool:
        return self.stopped

    # --- управление ---
    def pause(self) -> None:
        self.paused = True
        if self.sink:
            self.sink.pause()
        self._status()

    def resume(self) -> None:
        self.paused = False
        if self.sink:
            self.sink.resume()
        self._status()

    def stop(self) -> None:
        """Звук обрывается сразу; синтез бросает работу на ближайшем стыке."""
        self.stopped = True
        if self.sink:
            self.sink.stop()
        with self._cv:
            self._cv.notify_all()
        self._status()

    def wait(self, timeout: float | None = None) -> bool:
        """Ждать, пока всё прозвучит (или стоп)."""
        self._worker.join(timeout)
        if self.sink is None or self.stopped:
            return True
        return self.sink.done.wait(timeout)

    def audio(self) -> np.ndarray:
        """Весь сырой звук (частота движка) - то же, что вернул синтез."""
        return np.concatenate(self.raw) if self.raw else np.zeros(0, dtype=np.float32)

    @property
    def stalls(self) -> int:
        return self.sink.stalls if self.sink else 0

    # --- внутреннее ---
    def _rest(self) -> list[tuple[float, float]]:
        out = []
        for j, (chars, _gap) in enumerate(self.plan_items[self.fed:], start=self.fed):
            # паузу между кусками не считаем: её съедает тишина на краях кусков
            # (_join), и с ней звучание выходило бы длиннее, чем на деле
            d = self.speed.audio_sec(chars, self.tempo)
            # первые два куска идут без упреждения (первый проверяется один, см.
            # _synth_chunks) - их счёт ни с чем не перекрывается (замер 3.4)
            out.append((self.speed.synth_sec(d, overlapped=j >= 2), d))
        return out

    def wait_sec(self) -> float:
        elapsed = time.monotonic() - self._t_last if self._t_last else 0.0
        return wait_needed(self.ready_sec, self._rest(), elapsed, self.margin)

    def _start(self) -> None:
        with self._start_lock:
            if self.started or self.sink is None or self.stopped or not self._pending:
                return
            self.started = True
            self.first_sound = time.monotonic() - self.t0
            if self.paused:
                self.sink.pause()
            for a in self._pending:
                self.sink.write(a)
            self._pending = []
            self.sink.open(self.out_sr)

    def playing_chunk(self) -> int | None:
        """Номер куска, который сейчас звучит (кусок k уходит вместе с паузой
        перед ним), или None - звук ещё не начался или кончился."""
        if not self.started or self.sink is None:
            return None
        pos, acc = self.sink.played, 0
        for k, n in enumerate(self._out_lens):
            acc += n
            if pos < acc:
                return k if k < len(self.plan_items) else None
        return None

    def _put(self, a: np.ndarray) -> None:
        self._out_lens.append(len(a))
        with self._start_lock:
            if not self.started:
                self._pending.append(a)
                return
        self.sink.write(a)

    @property
    def out_sr(self) -> int:
        return OUT_SR if (self.hd or self.fx_chain) else int(self.sr or OUT_SR)
    # fx_chain None - «поправки самого голоса»: их подставит speak()

    def _polish(self, seg: np.ndarray) -> np.ndarray:
        sr = int(self.sr or OUT_SR)
        if not (self.hd or self.fx_chain):
            return seg
        n_out = int(round(len(seg) * OUT_SR / sr))
        if not np.any(seg):
            return np.zeros(n_out, dtype=np.float32)
        from .assemble import _chain, _ff
        chain = _chain(self.hd, self.fx_chain)
        if "aresample" not in chain.split(",")[-1]:
            chain += f",aresample={OUT_SR}"
        p = subprocess.run([_ff(), "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1",
                            "-i", "-", "-af", chain, "-f", "f32le", "-ar", str(OUT_SR),
                            "-ac", "1", "-"],
                           input=np.asarray(seg, dtype=np.float32).tobytes(),
                           capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype=np.float32).copy()

    def _run(self):
        try:
            while True:
                with self._cv:
                    while not self._out and not self.finished and not self.stopped:
                        self._cv.wait(0.5)
                    if self.stopped:
                        return
                    if not self._out and self.finished:
                        break
                    seg = self._out.popleft()
                out = self._polish(seg)
                self.ready_sec += len(seg) / float(self.sr or OUT_SR)
                if self.sink is not None and not self.stopped:
                    self._put(out)
                    if not self.started and (not self.gate or self.wait_sec() <= 0):
                        self._start()
                self._status()
            if self.sink is not None and not self.stopped:
                self._start()                 # всё готово - играем, сколько бы ни было
                if not self.started:
                    self.sink.done.set()      # звука нет вовсе - и играть нечего
                self.sink.finish()
        except Exception as e:                # ffmpeg, звуковая карта
            self.error = e
            if self.sink is not None:
                self.sink.stop()
        finally:
            self._status()

    def _tick(self):
        while not (self.stopped or (self.finished and not self._worker.is_alive())):
            time.sleep(0.25)
            if not self.started:
                # текущий кусок досчитывается - старт может наступить и до него
                if self._pending and self.wait_sec() <= 0:
                    self._start()
                self._status()

    def state(self) -> dict:
        if self.stopped:
            st = "stopped"
        elif self.error is not None:
            st = "error"
        elif not self.started:
            st = "wait"
        elif self.finished and not self._worker.is_alive():
            st = "done" if self.sink is None or self.sink.done.is_set() else "play_all"
        else:
            st = "play"
        total = self.ready_sec + sum(d for _s, d in self._rest())
        return {"state": st, "wait": self.wait_sec() if st == "wait" else 0.0,
                "ready": self.ready_sec, "total": total, "chunks": len(self.plan_items),
                "fed": self.fed, "paused": self.paused, "stalls": self.stalls,
                "played": self.sink.position if self.sink else 0.0,
                "first_sound": self.first_sound}

    def _status(self):
        if self.on_status:
            try:
                self.on_status(self.state())
            except Exception:
                pass                          # окно закрылось - не мешаем синтезу


def status_text(s: dict) -> str:
    """Строка для окна: «Звук начнётся через ~N с» → «Играет · считаю дальше»."""
    if s["state"] == "wait":
        if s["chunks"] == 0:
            return "Готовлю текст…"
        return f"Звук начнётся через ~{max(1, round(s['wait']))} с"
    if s["state"] == "play":
        return ("Пауза · считаю дальше" if s["paused"] else "Играет · считаю дальше")
    if s["state"] == "play_all":
        return "Пауза · всё посчитано" if s["paused"] else "Играет · всё посчитано"
    if s["state"] == "stopped":
        return "Остановлено"
    if s["state"] == "error":
        return "Звук потоком не вышел - подробности в журнале"
    return "Готово"


def _letters(t: str) -> int:
    return sum(1 for c in t if c.isalnum())


# --- куски: первый короткий ----------------------------------------------
def split_first(chunks: list[tuple], first_max: int = FIRST_MAX,
                first_min: int = FIRST_MIN) -> list[tuple]:
    """Отрезать от первого куска первое предложение (FIRST_MIN..FIRST_MAX знаков).

    Режем только по границе предложения, отмеченной при нарезке (отметка
    паузы внутри куска), - пауза там та же, что была бы внутри куска.
    """
    if not chunks:
        return chunks
    text, gap, mul, marks = (list(chunks[0]) + [()])[:4]
    if len(text) <= first_max:
        return chunks
    cut = [(p, g) for p, g in marks if first_min <= p <= first_max]
    if not cut:
        return chunks
    p, g = cut[-1]
    head, tail = text[:p].rstrip(), text[p:].lstrip()
    shift = len(text) - len(tail)
    hm = tuple(m for m in marks if m[0] < p)
    tm = tuple((q - shift, gg) for q, gg in marks if q > p)
    # замедление заголовка (если он склеен с первой фразой) остаётся у головы
    return [(head, g, mul, hm), (tail, gap, 1.0, tm)] + list(chunks[1:])


def choose_first(chunks: list[tuple], live: "Live | None") -> list[tuple]:
    """Отрезать короткий первый кусок - только если звук от этого начнётся раньше.

    Короткий первый кусок хорош, когда движок быстрее реального времени (Silero):
    звук идёт сразу после него. У клона F5 на каждый кусок уходит постоянная
    часть (прогон образца, около 5 с на M4, замер 3.4), и при RTF больше 1
    лишний кусок только отодвигает момент, когда можно играть без запинок.
    Решает расчёт старта по замеренной скорости этого голоса.
    """
    if live is None:
        return chunks
    alt = split_first(chunks)
    if len(alt) == len(chunks):
        return chunks
    if not live.gate:                   # «начинать сразу» - звук после первого куска
        return alt

    def start(ch):
        rest = []
        for c in ch:
            d = live.speed.audio_sec(_letters(c[0]))
            rest.append((live.speed.synth_sec(d), d))
        return wait_needed(0.0, rest, 0.0, live.margin) if rest else 0.0
    return alt if start(alt) < start(chunks) else chunks


# --- откуда в тексте каждый кусок (подсветка во вкладке) ---------------
def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Предложения исходного текста: [(начало, конец)] по абзацам."""
    import re
    from razdel import sentenize
    out = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        for sp in sentenize(m.group(0)):
            if sp.text.strip():
                out.append((m.start() + sp.start, m.start() + sp.stop))
    return out


def prepare_by_sentences(text: str, prep) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Текст для синтеза по предложениям: (подготовленный текст, места
    предложений в исходном, сколько букв в каждом подготовленном). Абзацы -
    как в исходном (пустая строка), чтобы паузы были те же."""
    spans = sentence_spans(text)
    paras, letters, last_end = [], [], None
    for a, b in spans:
        t = prep(text[a:b])
        if last_end is not None and "\n\n" in text[last_end:a].replace("\r", "").replace(" ", ""):
            paras.append("\n\n")
        elif paras:
            paras.append(" ")
        paras.append(t)
        letters.append(_letters(t))
        last_end = b
    return "".join(paras), spans, letters


def chunk_spans(chunks: list[tuple], spans: list[tuple[int, int]],
                letters: list[int]) -> list[tuple[int, int]]:
    """Место каждого куска в исходном тексте: по числу букв (нарезка на куски
    добавляет только знаки и пробелы, буквы и цифры не трогает)."""
    out, si, used = [], 0, 0
    for c in chunks:
        need = _letters(c[0])
        first = min(si, len(spans) - 1)
        while need > 0 and si < len(spans):
            take = min(need, letters[si] - used)
            need -= take
            used += take
            if used >= letters[si]:
                si, used = si + 1, 0
        last = max(first, min(si if used else si - 1, len(spans) - 1))
        out.append((spans[first][0], spans[last][1]) if spans else (0, 0))
    return out


# --- озвучить произвольный текст -----------------------------------------
def speak(text: str, *, engine: str, voice: str, live: Live | None,
          eng_kwargs: dict | None = None, language: str | None = None,
          prepared: bool = False, sound_check: bool = False, opts=None,
          use_cache: bool = True, lock_timeout: float | None = None,
          on_prepared=None, tempo: float | None = None, on_spans=None):
    """Озвучить текст потоком. Возвращает (сырой звук, частота).

    Ударения, числа, паузы, темп голоса - как в книге. Проверка слов по звуку
    по умолчанию выключена: это прослушивание, а не готовая книга (+19 %).
    """
    from . import pipeline as P
    from .normalize import detect_language
    from .engines.f5 import default_voices_dir
    from .voicefx import load as load_fx

    opts = opts or P.ConvertOptions()
    lang = language or detect_language(text or "")
    kw = dict(eng_kwargs or {})
    if engine in ("f5mlx", "f5", "f5torch") and "language" not in kw:
        kw["language"] = lang
    try:
        fx = load_fx(default_voices_dir(), voice)
    except Exception:
        fx = None                       # поправки голоса не прочитались - как есть
    # темп - не параметр движка (иначе другой ключ движка и перезагрузка модели
    # 1,3 ГБ на каждое движение ползунка), а множитель на кусок: у клона - в
    # модель (больше или меньше звука), у Silero - atempo после синтеза
    if tempo is None:
        tempo = float(kw.pop("speed", None) or (fx.speed if fx is not None else 1.0))
    if live is not None:
        if live.fx_chain is None:
            live.fx_chain = fx.filter_chain() if fx is not None else ""
        if engine == "silero":
            live.fx_chain = with_tempo(live.fx_chain, tempo)
        else:
            live.tempo = tempo
    if not ENGINE_LOCK.acquire(timeout=-1 if lock_timeout is None else lock_timeout):
        raise Busy("Herald сейчас занят другой озвучкой")
    try:
        eng = P._loaded_engine(engine, kw)
        sr = eng.sample_rate
        prep = (lambda x: x) if prepared else (
            lambda x: P._prepare_text(x, getattr(eng, "stress_format", None),
                                      opts.expand_numbers, opts.use_stress, lang))
        if on_spans is not None:            # вкладка: подсветка звучащего предложения
            t, sp, lt = prepare_by_sentences(text, prep)
        else:
            t = prep(text)
        if on_prepared:
            on_prepared()
        chunks = P._chunk_paragraphs(t, eng.max_chunk_chars, opts.gap_sec,
                                     opts.para_gap_sec, opts.dialog_gap_sec)
        chunks = choose_first(chunks, live)
        if on_spans is not None:
            on_spans(chunk_spans(chunks, sp, lt))
        try:
            # края у готовых голосов Silero не проверяем: они их не съедают, а
            # пересчёт дал бы тот же звук - распознавание только задержало бы звук
            audio = P._synth_chunks(eng, voice, chunks, sr, opts.gap_sec, lang=lang,
                                    use_cache=use_cache, sound_check=sound_check,
                                    strict_words=False, live=live,
                                    check_edges=P._clone(eng))
        except BaseException:
            if live is not None:
                live.stop()
            raise
        return audio, sr
    finally:
        ENGINE_LOCK.release()


def with_tempo(chain: str, tempo: float) -> str:
    """Поправки голоса (chain - без темпа) с темпом впереди: готовый голос
    Silero меняет темп после синтеза (atempo), клон - в самой модели."""
    t = f"atempo={tempo:.3f}" if abs(tempo - 1.0) > 1e-3 else ""
    return ",".join(x for x in (t, chain or "") if x)


class Busy(RuntimeError):
    """Движок занят другим заказом."""


# Постоянная часть счёта куска у клона F5 на M4 (12 шагов): прогон образца.
# Замер 3.4: 21 знак (1,9 с звука) - 6,2 с, 187 знаков (13,4 с) - 14,1 с.
CLONE_FIXED_SEC = 5.3


def make_speed(engine: str, eng_kwargs: dict | None, voice: str, sound_check: bool,
               prior_rtf: float, accel: str = "mlx") -> Speed:
    """Скорость этого движка с этим голосом: замеры, а до них - прикидка профиля."""
    from . import profiles
    fixed = 0.0 if engine == "silero" else CLONE_FIXED_SEC * profiles.ACCEL_MUL.get(accel, 1.0)
    return Speed(speed_key(engine, eng_kwargs, voice, sound_check), prior_rtf,
                 prior_fixed=fixed)


def speed_key(engine: str, eng_kwargs: dict | None, voice: str, sound_check: bool) -> str:
    kw = ",".join(f"{k}={v}" for k, v in sorted((eng_kwargs or {}).items())
                  if k in ("steps", "variant", "schedule", "language"))
    return f"{engine}|{kw}|{voice}|{'z' if sound_check else '-'}"
