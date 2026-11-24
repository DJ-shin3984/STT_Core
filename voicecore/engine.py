"""Voice Core engine.

API (fixed; Shell depends on it):
    eng = Engine(cfg, on_event)        # cfg = tomllib.load(config.toml); on_event(Event) is called from engine threads
    eng.load()                         # build adapters (registry), vocab, output, log; warm up
    utt_id = eng.start(mode, source=None)   # PTT press. mode "verbatim"|"polished". source None = mic,
                                            # else an audio source with .read() -> np.float32 chunk or None at EOF
    eng.stop()                         # PTT release: flush 1st pass, run 2nd pass + correction (+LLM) + output
                                       # in a worker thread; emits UtteranceFinalized then OutputDone
    eng.cancel()                       # drop the current utterance, no output
    eng.close()
Events: voicecore.events. One output per utt_id; a result past the deadline
(cfg["engine"]["deadline_verbatim_ms"] / deadline_polished_ms) is discarded with EngineError.

스레드: 상시 마이크 1개(load~close, 마이크 → 링버퍼 → 소비 중인 발화 큐),
발화마다 인식 1개(큐 → streaming.feed → TranscriptBuffer, stop 후 그대로 종료 경로)와
파일 소스일 때 캡처 1개, 엔진 공용 로그 1개. 마이크 발화는 preroll_ms 만큼 링버퍼 직전 구간을 앞에 붙인다.
"""
from __future__ import annotations

import dataclasses
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from . import correction
from .adapters import registry
from .audio import MicSource, RingBuffer, UtteranceBuffer
from .events import EngineError, OutputDone, PartialUpdated, SegmentStable, UtteranceFinalized
from .log import CorrectionLog
from .output import ClipboardOutput
from .transcript import TranscriptBuffer
from .vocab import Vocabulary


def _ms(t0: float, t1: float) -> float:
    return round((t1 - t0) * 1000, 1)


class Engine:
    def __init__(self, cfg: dict, on_event):
        self.cfg, self.on_event = cfg, on_event
        self.sr = cfg["audio"]["sample_rate"]
        self.streaming = self.final = self.llm = None   # llm: 1-4에서 callable(text) -> text
        self._lock, self._start_lock = threading.Lock(), threading.Lock()
        self._stt_lock, self._final_lock = threading.Lock(), threading.Lock()
        self._cur = None
        self._threads: list[threading.Thread] = []
        self._last_ms = 0
        self._logq: queue.Queue = queue.Queue()
        self._log_thread = None
        self.mic = self._mic_u = None       # 상시 마이크와 지금 소비 중인 발화
        self._closing = threading.Event()

    # ---------- 이벤트 ----------
    def _emit(self, ev):
        try:
            self.on_event(ev)
        except Exception as e:  # 콜백 예외는 삼키고 EngineError로
            if not isinstance(ev, EngineError):
                self._emit(EngineError(getattr(ev, "utt_id", ""), f"on_event 예외: {e!r}"))

    # ---------- 수명 ----------
    def load(self):
        c, mdir = self.cfg, self.cfg["models"]["dir"]
        self.ring = RingBuffer(c["engine"]["ring_sec"], self.sr)
        try:                                # 마이크는 한 번 열어 close()까지 유지
            a = c["audio"]
            self.mic = MicSource(a["input_device"], self.sr, a["chunk_ms"])
            self.mic.start()
            th = threading.Thread(target=self._mic_loop, name="vc-mic", daemon=True)
            th.start()
            self._threads.append(th)
        except Exception as e:
            print(f"[engine] 경고: 마이크 열기 실패({e!r}) → 파일 소스만 사용", flush=True)
            self.mic = None
        self.vocab = Vocabulary.load(c["vocab"]["file"])
        hw = self.vocab.hotwords()
        t = time.perf_counter()
        self.streaming = registry.build_streaming(c["stt"]["streaming"], models_dir=mdir)
        self.streaming.load()
        print(f"[engine] streaming {c['stt']['streaming']['model']} load {_ms(t, time.perf_counter()):.0f}ms",
              flush=True)
        if c["engine"]["second_pass"]:
            t = time.perf_counter()
            self.final = registry.build_offline(c["stt"]["final"], models_dir=mdir)
            self.final.load()
            print(f"[engine] final {c['stt']['final']['model']} load {_ms(t, time.perf_counter()):.0f}ms",
                  flush=True)
        # 워밍업(무음 0.5초): 첫 발화 지연 제거, hotwords 인식기도 여기서 만든다
        t, z = time.perf_counter(), np.zeros(self.sr // 2, np.float32)
        self.streaming.start(hotwords=hw)
        self.streaming.feed(z, self.sr)
        self.streaming.finish()
        if self.final is not None:
            self.final.transcribe(z, self.sr, hotwords=hw)
        print(f"[engine] warmup {_ms(t, time.perf_counter()):.0f}ms", flush=True)
        self.output = ClipboardOutput(paste=c["output"]["paste"],
                                      restore_delay_ms=c["output"]["restore_delay_ms"])
        self.log = CorrectionLog(c["log"]["dir"], c["log"]["max_mb"])
        self._log_thread = threading.Thread(target=self._log_loop, name="vc-log", daemon=True)
        self._log_thread.start()

    def close(self):
        self.cancel()
        self._closing.set()
        for th in list(self._threads):
            th.join(timeout=30)
        if self._log_thread:
            self._logq.put(None)
            self._log_thread.join(timeout=30)
            self._log_thread = None
            if hasattr(self.log, "close"):   # 로그 구현의 자체 큐까지 비움
                self.log.close()
        if self.mic is not None:
            self.mic.stop()
            self.mic = None
        for a in (self.streaming, self.final):
            if a is not None:
                a.unload()
        self.streaming = self.final = None

    # ---------- PTT ----------
    def _new_id(self) -> str:
        n = max(time.time_ns() // 1_000_000, self._last_ms + 1)   # 시간 기반, 단조 증가
        self._last_ms = n
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(n / 1000)) + f"-{n % 1000:03d}"

    def start(self, mode: str, source=None) -> str:
        with self._start_lock:
            prev = self._cur
            if prev is not None:
                self.stop()                 # 녹음 중이면 끝내고(종료 경로는 자기 스레드에서 계속)
            if source is None and self.mic is None:
                raise RuntimeError("마이크 없음(load 때 열기 실패)")
            u = SimpleNamespace(id=self._new_id(), mode=mode, src=source, q=queue.Queue(),
                                buf=UtteranceBuffer(self.cfg["engine"]["utterance_max_sec"], self.sr),
                                stop_ev=threading.Event(), stopped=False, cancelled=False,
                                outputting=False, t_stop=None)
            u.recog = threading.Thread(target=self._recognize, args=(u,), name=f"vc-rec-{u.id}", daemon=True)
            ths = [u.recog]
            if source is not None:          # 파일 등: 발화마다 캡처 스레드
                ths.append(threading.Thread(target=self._capture, args=(u,), name=f"vc-cap-{u.id}", daemon=True))
            with self._lock:
                self._cur = u
                self._threads = [t for t in self._threads if t.is_alive()] + ths
                if source is None:          # 마이크: 링버퍼 직전 구간(프리롤)을 먼저 넣고 지금부터 소비
                    n = self.cfg["engine"]["preroll_ms"] * self.sr // 1000
                    pre = self.ring.pcm()[-n:] if n else []
                    if len(pre):
                        self._push(u, pre)
                    self._mic_u = u
            for t in ths:
                t.start()
            return u.id

    def stop(self):
        with self._lock:
            u = self._cur
            if u is None or u.stopped or u.cancelled:
                return None
            u.stopped, u.t_stop = True, time.perf_counter()
            u.stop_ev.set()
            self._detach(u)
            return u.id

    def cancel(self):
        with self._lock:
            u = self._cur
            if u is None or u.outputting or u.cancelled:
                return None
            u.cancelled = True
            u.stop_ev.set()
            self._detach(u)
            return u.id

    def _detach(self, u):                   # self._lock 안에서 호출
        if self._mic_u is u:
            self._mic_u = None
            u.q.put(None)

    # ---------- 스레드 ----------
    def _push(self, u, c):
        u.buf.append(c)
        u.q.put(c)

    def _mic_loop(self):                    # 상시: 마이크 → 링버퍼 (+ 소비 중인 발화)
        while not self._closing.is_set():
            c = self.mic.read(timeout=0.05)
            if c is None:
                continue
            with self._lock:
                self.ring.append(c)
                if self._mic_u is not None:
                    self._push(self._mic_u, c)

    def _capture(self, u):
        src = u.src
        try:
            src.start()
            while not u.stop_ev.is_set():
                c = src.read(timeout=0.05)
                if c is not None:
                    self._push(u, c)
                elif getattr(src, "eof", False):
                    break
            src.stop()
            while not u.cancelled and (c := src.read(timeout=0)) is not None:  # 마이크 큐 잔량
                self._push(u, c)
        except Exception as e:
            self._emit(EngineError(u.id, f"캡처 실패: {e!r}"))
        finally:
            u.q.put(None)

    def _recognize(self, u):
        try:
            flush, u.t_flush = None, 0.0
            # 2차 인식이 있으면 stop 뒤 1차 backlog·flush는 결과에 안 쓰이므로 건너뜀(지연 절약)
            skip = bool(self.cfg["engine"]["second_pass"] and self.final is not None)
            with self._stt_lock:            # 스트리밍 어댑터는 한 번에 한 발화
                if u.cancelled:
                    return
                self.streaming.start(hotwords=self.vocab.hotwords())
                tb = TranscriptBuffer(self.cfg["engine"]["stable_after"])
                stable_c = ""
                while (c := u.q.get()) is not None:
                    if u.cancelled:
                        return
                    if skip and u.stopped:
                        continue
                    p = self.streaming.feed(c, self.sr)
                    if p is None:
                        continue
                    stable, unstable, new = tb.update(p)
                    if new:                 # stable 구간에만 보정, unstable은 그대로
                        stable_c = correction.correct(stable, self.vocab).text
                    self._emit(PartialUpdated(u.id, stable_c, unstable))
                    if new:
                        self._emit(SegmentStable(u.id, new))
                u.stop_ev.wait()            # 파일 EOF 후에도 stop()까지 기다림
                if u.cancelled:
                    return
                if not skip:
                    t = time.perf_counter()
                    flush = self.streaming.finish()
                    u.t_flush = _ms(t, time.perf_counter())
            self._finalize(u, flush)
        except Exception as e:
            self._emit(EngineError(u.id, f"처리 실패: {e!r}"))
            self._emit(OutputDone(u.id, False, self.cfg["output"]["method"]))

    def _finalize(self, u, flush):
        c, mode, method = self.cfg, u.mode, self.cfg["output"]["method"]
        pcm, hw = u.buf.pcm(), self.vocab.hotwords()
        with self._final_lock:
            t0 = time.perf_counter()
            raw = (self.final.transcribe(pcm, self.sr, hotwords=hw).text
                   if flush is None else flush.text)
            t1 = time.perf_counter()
            fast = correction.correct(raw, self.vocab)
            t2 = time.perf_counter()
            text = fast.text
            if self.llm is not None and mode == "polished" and mode in c["llm"]["enabled_for"]:
                text = self.llm(text)       # 1-4 훅
            t3 = time.perf_counter()
            ok, reason = correction.safety_check(raw, text, self.vocab)
            if not ok:
                text = fast.text
            timings = dict(flush=u.t_flush, stt_final=round(u.t_flush + _ms(t0, t1), 1), correction=_ms(t1, t2), llm=_ms(t2, t3))
            deadline = c["engine"].get(f"deadline_{mode}_ms", 0)
            record = dict(ts=time.time(), mode=mode, raw=raw, fast=fast.text, final=text,
                          corrections=[dataclasses.asdict(x) for x in fast.corrections],
                          timings_ms=timings, truncated=u.buf.truncated)
            with self._lock:
                if u.cancelled:
                    return
                late = deadline and _ms(u.t_stop, time.perf_counter()) > deadline
                u.outputting = not late     # 이후 cancel() 무시, 출력은 이 한 번뿐
            if late:
                timings["total"] = _ms(u.t_stop, time.perf_counter())
                self._emit(EngineError(u.id, f"deadline {deadline}ms 초과({timings['total']:.0f}ms): 결과 폐기"))
                self._emit(OutputDone(u.id, False, method))
                self._logq.put((u.id, pcm, dict(record, final="", error="deadline")))
                return
            t4 = time.perf_counter()
            sent = bool(text) and (method != "clipboard" or self.output.send(text))
            t5 = time.perf_counter()
            timings.update(output=_ms(t4, t5), total=_ms(u.t_stop, t5))
            self._emit(UtteranceFinalized(u.id, mode, raw, text, timings))
            self._emit(OutputDone(u.id, bool(sent), method))
            self._logq.put((u.id, pcm, record))

    def _log_loop(self):
        while (item := self._logq.get()) is not None:
            try:
                self.log.write(item[0], item[1], self.sr, item[2])
            except Exception as e:
                self._emit(EngineError(item[0], f"로그 실패: {e!r}"))
