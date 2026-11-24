"""오디오 소스(마이크/파일)와 버퍼. 모든 소스는 float32 mono 1-D 청크를 돌려준다."""
from __future__ import annotations

import collections
import queue
import time

import numpy as np


def _device(spec):
    """"" = 기본, 숫자 문자열 = 인덱스, 그 외 = 입력 장치 이름 부분 일치."""
    spec = str(spec or "").strip()
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and spec.lower() in d["name"].lower():
            return i
    raise ValueError(f"입력 장치 없음: {spec}")


class MicSource:
    def __init__(self, device="", sr: int = 16000, chunk_ms: int = 20):
        self.device, self.sr, self.block = _device(device), sr, sr * chunk_ms // 1000
        self.q: queue.Queue = queue.Queue()
        self.stream = None

    def _cb(self, indata, frames, t, status):  # 오디오 콜백: 복사만 하고 큐로
        self.q.put(indata[:, 0].copy())

    def start(self):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                     blocksize=self.block, device=self.device, callback=self._cb)
        self.stream.start()

    def read(self, timeout=None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):  # 큐에 남은 청크는 read()로 계속 읽을 수 있다
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


class FileSource:
    def __init__(self, path, chunk_ms: int = 20, realtime: bool = False, sr: int = 16000):
        import soundfile as sf
        pcm, fsr = sf.read(str(path), dtype="float32", always_2d=True)
        if fsr != sr:
            raise ValueError(f"{path}: {fsr}Hz (필요: {sr}Hz, 리샘플링 안 함)")
        self.pcm, self.sr, self.block = pcm.mean(axis=1).astype(np.float32), sr, sr * chunk_ms // 1000
        self.realtime, self.pos, self.t0, self.eof = realtime, 0, None, False

    def start(self):
        self.t0 = time.perf_counter()

    def read(self, timeout=None):
        if self.eof or self.pos >= len(self.pcm):
            self.eof = True
            return None
        if self.realtime:  # 시작 시각 기준 절대 일정 → 소비가 늦어도 누적 지연 없음
            if self.t0 is None:
                self.t0 = time.perf_counter()
            wait = self.t0 + (self.pos + self.block) / self.sr - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        c = self.pcm[self.pos:self.pos + self.block]
        self.pos += self.block
        return c

    def stop(self):
        self.eof = True


class UtteranceBuffer:
    """발화 전체 PCM. 상한을 넘으면 뒤를 버리고 truncated=True."""

    def __init__(self, max_sec: float, sr: int = 16000):
        self.max, self.n, self.chunks, self.truncated = int(max_sec * sr), 0, [], False

    def append(self, chunk):
        room = self.max - self.n
        if len(chunk) > room:
            self.truncated, chunk = True, chunk[:max(room, 0)]
        if len(chunk):
            self.chunks.append(chunk)
            self.n += len(chunk)

    def pcm(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)


class RingBuffer:
    """최근 N초 보관(2단계용)."""

    def __init__(self, sec: float, sr: int = 16000):
        self.max, self.n, self.chunks = int(sec * sr), 0, collections.deque()

    def append(self, chunk):
        self.chunks.append(chunk)
        self.n += len(chunk)
        while self.chunks and self.n - len(self.chunks[0]) >= self.max:
            self.n -= len(self.chunks.popleft())

    def pcm(self) -> np.ndarray:
        a = np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)
        return a[-self.max:] if self.max else a[:0]
