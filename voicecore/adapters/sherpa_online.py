"""sherpa-onnx 스트리밍 transducer 어댑터 (Nemotron 3.5 ASR Streaming, zipformer-ko).

hotwords: start(hotwords=[...]) 시 임시 hotwords_file + modified_beam_search 로
인식기를 다시 만든다(같은 목록이면 재사용). 빌드가 거부하면 greedy 로 폴백하고
meta["biasing"]="unsupported_in_this_build".
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

from .base import StreamingSTT, STTResult

_BEAM_OK: dict[str, bool] = {}  # 모델 폴더 → modified_beam_search 가능 여부


def _bpe_vocab(d: Path) -> str:
    """bpe.model(sentencepiece protobuf) → bpe.vocab(piece, score 탭 구분). hotword 인코딩용.
    sentencepiece 패키지 없이 ModelProto 앞쪽의 pieces(field 1)만 읽는다."""
    out = d / "bpe.vocab"
    if out.exists():
        return str(out)
    b, i, lines = (d / "bpe.model").read_bytes(), 0, []

    def varint(j):
        n = sh = 0
        while True:
            n, sh, j = n | (b[j] & 0x7F) << sh, sh + 7, j + 1
            if b[j - 1] < 0x80:
                return n, j
    while b[i] == 0x0A:  # pieces { 1: piece(str), 2: score(float), 3: type }
        ln, i = varint(i + 1)
        n, j = varint(i + 1)  # sub[0] == 0x0A
        piece = b[j:j + n].decode()
        score = float(np.frombuffer(b[j + n + 1:j + n + 5], "<f4")[0]) if b[j + n] == 0x15 else 0.0
        lines.append(f"{piece}\t{score}")
        i += ln
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out)


def _pick(d: Path, stem: str) -> str:
    # int8 우선, 없으면 fp32
    c = sorted(d.glob(f"{stem}*.int8.onnx")) or sorted(d.glob(f"{stem}*.onnx"))
    if not c:
        raise FileNotFoundError(f"{d}/{stem}*.onnx 없음")
    return str(c[0])


class SherpaOnlineSTT(StreamingSTT):
    name = "sherpa_online"

    def __init__(self, model_dir: str, provider: str = "cpu", language: str = "ko",
                 decoding: str = "greedy_search", hotwords_score: float = 2.0, num_threads: int = 4):
        self.dir = Path(model_dir)
        self.provider, self.language, self.decoding = provider, language, decoding
        self.hotwords_score, self.num_threads = hotwords_score, num_threads
        self.nemotron = "nemotron" in self.dir.name
        self.rec = self.stream = None
        self._hw: tuple = ()          # 현재 인식기에 들어간 hotwords
        self.biasing = "off"

    def _build(self, decoding: str, hotwords_file: str = ""):
        d = self.dir
        kw = {}
        if hotwords_file and (d / "bpe.model").exists():  # 한글 구절 → bpe 토큰
            kw = dict(modeling_unit="bpe", bpe_vocab=_bpe_vocab(d))
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"), encoder=_pick(d, "encoder"),
            decoder=_pick(d, "decoder"), joiner=_pick(d, "joiner"),
            num_threads=self.num_threads, provider=self.provider,
            decoding_method=decoding, hotwords_file=hotwords_file,
            hotwords_score=self.hotwords_score,
            model_type="nemotron" if self.nemotron else "", **kw)

    def load(self) -> None:
        if self.decoding != "greedy_search" and not self._beam_ok():
            print(f"[sherpa_online] {self.decoding} 미지원 빌드 → greedy_search")
            self.decoding = "greedy_search"
        self.rec = self._build(self.decoding)

    def _beam_ok(self) -> bool:
        """Nemotron + sherpa-onnx 1.13.8 은 modified_beam_search 에서 C++ 가 프로세스를 종료시켜
        예외로 잡을 수 없다 → 하위 프로세스로 한 번 확인하고 결과를 캐시한다(약 8초)."""
        if not self.nemotron:
            return True
        key = str(self.dir.resolve())
        if key not in _BEAM_OK:
            d = self.dir
            code = ("import sys, sherpa_onnx as s; a = sys.argv[1:]; s.OnlineRecognizer.from_transducer("
                    "tokens=a[0], encoder=a[1], decoder=a[2], joiner=a[3], "
                    "decoding_method='modified_beam_search', model_type='nemotron')")
            r = subprocess.run([sys.executable, "-I", "-c", code, str(d / "tokens.txt"),
                                _pick(d, "encoder"), _pick(d, "decoder"), _pick(d, "joiner")],
                               capture_output=True, timeout=300)
            _BEAM_OK[key] = r.returncode == 0
        return _BEAM_OK[key]

    def _set_hotwords(self, hotwords: list[str]) -> None:
        if tuple(hotwords) == self._hw:
            return
        self._hw, was = tuple(hotwords), self.biasing
        if not hotwords:
            if was == "hotwords":  # hotword 인식기 → 원래 설정으로
                self.rec = self._build(self.decoding)
            self.biasing = "off"
            return
        if not self._beam_ok():  # 지금 인식기(greedy) 그대로 사용
            print("[sherpa_online] modified_beam_search 미지원 빌드 → hotwords 없이 greedy")
            self.biasing = "unsupported_in_this_build"
            return
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="hotwords-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(hotwords) + "\n")   # 한 줄에 한 구절, 선택적 ":score"
        try:
            self.rec, self.biasing = self._build("modified_beam_search", path), "hotwords"
        except Exception as e:  # 파이썬 예외로 오는 거부
            print(f"[sherpa_online] hotwords 거부 → greedy 유지: {e}")
            self.biasing = "unsupported_in_this_build"
        finally:
            os.unlink(path)

    def start(self, hotwords: list[str] | None = None) -> None:
        self._set_hotwords(list(hotwords or []))
        self.stream = self.rec.create_stream()
        if self.nemotron and self.language:
            self.stream.set_option("language", self.language)
        self.sr, self.samples, self.last = 16000, 0, ""

    def _decode(self) -> str:
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return "".join(self.rec.get_result_all(self.stream).tokens).strip()  # 토큰 앞 공백 = 띄어쓰기

    def feed(self, pcm: np.ndarray, sr: int) -> str | None:
        self.sr = sr
        self.samples += len(pcm)
        self.stream.accept_waveform(sr, np.asarray(pcm, dtype=np.float32))
        text = self._decode()
        if text != self.last:
            self.last = text
            return text
        return None

    def finish(self) -> STTResult:
        t0 = time.perf_counter()
        self.stream.accept_waveform(self.sr, np.zeros(int(0.5 * self.sr), np.float32))  # 꼬리 패딩
        self.stream.input_finished()
        text = self._decode()
        ms = (time.perf_counter() - t0) * 1000
        return STTResult(text, ms, self.samples / self.sr, meta=dict(
            adapter=self.name, model=self.dir.name, provider=self.provider, biasing=self.biasing))

    def unload(self) -> None:
        self.rec = self.stream = None
