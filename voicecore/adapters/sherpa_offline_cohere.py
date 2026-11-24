"""sherpa-onnx Cohere Transcribe 오프라인 어댑터 (OfflineRecognizer.from_cohere_transcribe).

- 언어: 필수(ISO "ko"). 빈 값이면 모델이 "Please specify a language" 로 빈 결과를 낸다(자동 감지 없음).
- hotwords/context: 미지원 → 무시, meta["biasing"]="unsupported".
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sherpa_onnx

from .base import OfflineSTT, STTResult


class SherpaCohereSTT(OfflineSTT):
    name = "sherpa_offline_cohere"

    def __init__(self, model_dir: str, provider: str = "cpu", language: str = "ko",
                 num_threads: int = 4):
        self.dir, self.provider, self.num_threads = Path(model_dir), provider, num_threads
        self.language = language
        self.rec = None

    def load(self) -> None:
        d = self.dir
        self.rec = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
            encoder=str(d / "encoder.int8.onnx"), decoder=str(d / "decoder.int8.onnx"),
            tokens=str(d / "tokens.txt"), num_threads=self.num_threads,
            language=self.language, provider=self.provider)

    def transcribe(self, pcm: np.ndarray, sr: int,
                   hotwords: list[str] | None = None, context: str = "") -> STTResult:
        t0 = time.perf_counter()
        s = self.rec.create_stream()
        s.accept_waveform(sr, np.asarray(pcm, dtype=np.float32))
        self.rec.decode_stream(s)
        ms = (time.perf_counter() - t0) * 1000
        meta = dict(adapter=self.name, model=self.dir.name, provider=self.provider,
                    biasing="unsupported", language=self.language)
        return STTResult(s.result.text.strip(), ms, len(pcm) / sr, meta=meta)

    def unload(self) -> None:
        self.rec = None
