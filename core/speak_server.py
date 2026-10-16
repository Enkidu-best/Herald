"""Озвучка для других программ (3.4): локальный сервис /speak.

Слушает только 127.0.0.1, по умолчанию выключен (Настройки → «Озвучка для
других программ» или `cli.py --serve`). Без интернета и облака.

    POST /speak  {"text": "...", "voice": "...", "fast": true, "play": false}
        play=false (по умолчанию) - ответ: звук потоком, WAV (PCM 16 бит, моно,
        48 кГц), кусок за куском по мере готовности (chunked). Начало ответа
        придерживается по тому же расчёту, что в окне: если играть звук сразу,
        как он приходит, он не будет запинаться.
        play=true - Herald играет сам, ответ сразу: {"ok": true}.
        fast=true (по умолчанию) - готовый голос Silero (voice - его имя, иначе
        «aidar»); fast=false - клон (voice - имя образца, иначе выбранный в окне).
    POST /stop   - остановить звук и синтез.
    GET  /health - {"ok": true, "version": "3.4"}.

Запросы из браузера пускаются только со страниц localhost/127.0.0.1, а Host
должен быть 127.0.0.1 или localhost: чужой сайт в браузере не сможет заставить
Herald говорить (и подмена DNS - тоже).
"""

from __future__ import annotations

import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

DEFAULT_PORT = 8716
DEFAULT_IDLE_MIN = 10
FAST_VOICE = "aidar"
LOCAL_HOSTS = ("127.0.0.1", "localhost")


def wav_header(sr: int) -> bytes:
    """Заголовок WAV «неизвестной длины» - для звука потоком."""
    size = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", size) + b"WAVE" + b"fmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", size))


def pcm16(a: np.ndarray) -> bytes:
    return (np.clip(a, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class SpeakServer:
    """Сервис в фоне. default_voice() - голос клона, если в запросе не указан."""

    def __init__(self, port: int = DEFAULT_PORT, idle_min: float = DEFAULT_IDLE_MIN,
                 default_voice=None, profile_kwargs=None, log=None):
        self.port, self.idle_min = int(port), float(idle_min)
        self.default_voice = default_voice or (lambda: None)
        self.profile_kwargs = profile_kwargs or (lambda: {})
        self.log = log or (lambda msg: None)
        self._httpd: ThreadingHTTPServer | None = None
        self._job = None                     # текущий Live
        self._job_lock = threading.Lock()
        self.last_use = time.monotonic()
        self._loaded = False
        self._stop_idle = threading.Event()

    # --- жизнь сервиса ---
    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        srv = self

        class H(_Handler):
            server_ref = srv
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True,
                         name="speak-http").start()
        self._stop_idle.clear()
        threading.Thread(target=self._idle_loop, daemon=True, name="speak-idle").start()
        threading.Thread(target=self._warm, daemon=True, name="speak-warm").start()
        self.log(f"озвучка для других программ: {self.url}")

    def stop(self) -> None:
        self.stop_job()
        h, self._httpd = self._httpd, None
        self._stop_idle.set()
        if h is not None:
            h.shutdown()
            h.server_close()

    def _warm(self) -> None:
        """Быстрый голос и ударения - заранее: первый ответ ≤ 1 с."""
        try:
            from . import pipeline as P
            from .stream import ENGINE_LOCK
            with ENGINE_LOCK:
                eng = P._loaded_engine("silero", {})
                P._prepare_text("Проверка связи.", eng.stress_format, True, True, "ru")
                eng.synth_chunk("Пров+ерка.", FAST_VOICE)
            self._loaded = True
        except Exception as e:
            self.log(f"сервис: быстрый голос не загрузился заранее: {e}")

    def _idle_loop(self) -> None:
        """Модели держим, пока сервис включён; после простоя - выгружаем."""
        while not self._stop_idle.wait(20):
            if (self._loaded and time.monotonic() - self.last_use > self.idle_min * 60
                    and self._job is None):
                from . import pipeline as P
                from .stream import ENGINE_LOCK
                if ENGINE_LOCK.acquire(timeout=0):
                    try:
                        P.unload_engines()
                        self._loaded = False
                        self.log("сервис: модели выгружены после простоя")
                    finally:
                        ENGINE_LOCK.release()

    # --- заказы ---
    def stop_job(self) -> None:
        with self._job_lock:
            job, self._job = self._job, None
        if job is not None:
            job.stop()

    def run_job(self, req: dict, sink) -> object:
        """Озвучить запрос в sink. Возвращает Live (после конца звука)."""
        from .stream import Live, make_speed, speak
        from . import profiles
        text = str(req.get("text") or "").strip()
        if not text:
            raise ValueError("пустой text")
        fast = bool(req.get("fast", True))
        voice = req.get("voice")
        if fast:
            engine, kw = "silero", {}
            from .engines.silero import VOICES
            if voice not in {v.id for v in VOICES}:
                voice = FAST_VOICE
            prior = profiles.SILERO.rtf
        else:
            from .engines import clone_engine
            engine, kw = clone_engine(), dict(self.profile_kwargs())
            voice = voice or self.default_voice()
            if not voice:
                raise ValueError("нет голоса: укажите voice или выберите голос в окне")
            prior = 1.8                      # до первого замера - как клон на Mac
        self.stop_job()
        from .engines import accel_label
        live = Live(sink, make_speed(engine, kw, voice, False, prior, accel_label()))
        with self._job_lock:
            self._job = live
        self.last_use = time.monotonic()
        try:
            speak(text, engine=engine, voice=voice, live=live, eng_kwargs=kw,
                  lock_timeout=30)
            self._loaded = True
        finally:
            self.last_use = time.monotonic()
        return live

    def finish_job(self, live) -> None:
        with self._job_lock:
            if self._job is live:
                self._job = None


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_ref: SpeakServer = None

    def log_message(self, fmt, *args):
        pass                                 # без шума в консоли

    # --- безопасность: только свой компьютер ---
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in LOCAL_HOSTS

    def _origin(self) -> str | None:
        o = self.headers.get("Origin")
        if not o:
            return None
        return o if (urlparse(o).hostname or "") in LOCAL_HOSTS else ""

    def _cors(self):
        o = self._origin()
        if o:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Expose-Headers", "X-Herald-Sample-Rate")

    def _deny(self) -> bool:
        if not self._host_ok() or self._origin() == "":
            self._json(403, {"ok": False, "error": "только для программ на этом компьютере"})
            return True
        return False

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        if self._deny():
            return
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self._deny():
            return
        if self.path.rstrip("/") == "/health":
            from . import VERSION
            self._json(200, {"ok": True, "version": VERSION})
        else:
            self._json(404, {"ok": False, "error": "нет такого адреса"})

    def do_POST(self):
        if self._deny():
            return
        srv = self.server_ref
        path = self.path.rstrip("/")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"ok": False, "error": "тело запроса - JSON"})
            return
        if path == "/stop":
            srv.stop_job()
            self._json(200, {"ok": True})
            return
        if path != "/speak":
            self._json(404, {"ok": False, "error": "нет такого адреса"})
            return
        if not str(req.get("text") or "").strip():
            self._json(400, {"ok": False, "error": "пустой text"})
            return
        from .pipeline import Cancelled
        from .stream import Busy, CallbackSink, OUT_SR, default_sink
        if req.get("play"):
            sink = default_sink()
            if sink is None:
                self._json(500, {"ok": False, "error": "нет звукового устройства"})
                return

            def bg():
                try:
                    live = srv.run_job(req, sink)
                    live.wait()
                    srv.finish_job(live)
                except Exception as e:
                    srv.log(f"сервис: озвучка не удалась: {e}")
            threading.Thread(target=bg, daemon=True, name="speak-play").start()
            self._json(200, {"ok": True})
            return

        state = {"head": False}

        def send(a: np.ndarray):
            if not state["head"]:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("X-Herald-Sample-Rate", str(OUT_SR))
                self.send_header("Cache-Control", "no-store")
                self._cors()
                self.end_headers()
                self._chunk(wav_header(OUT_SR))
                state["head"] = True
            self._chunk(pcm16(a))

        sink = CallbackSink(send)
        try:
            live = srv.run_job(req, sink)
            live.wait()
            srv.finish_job(live)
            if not state["head"]:
                send(np.zeros(0, dtype=np.float32))
            self.wfile.write(b"0\r\n\r\n")
        except Busy as e:
            if not state["head"]:
                self._json(503, {"ok": False, "error": str(e)})
        except Cancelled:                    # новый /speak или /stop
            if state["head"]:
                self.wfile.write(b"0\r\n\r\n")
            else:
                self._json(409, {"ok": False, "error": "остановлено"})
        except ValueError as e:
            if not state["head"]:
                self._json(400, {"ok": False, "error": str(e)})
        except Exception as e:
            if not state["head"]:
                self._json(500, {"ok": False, "error": str(e)})
            else:
                self.close_connection = True

    def _chunk(self, b: bytes):
        self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
        self.wfile.flush()
