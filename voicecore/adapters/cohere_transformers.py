"""Cohere Transcribe 03-2026 transformers 백엔드. 1-0 비교용.

**미검증**: transformers>=5.4 가 필요해 기본 venv 에서 실행 불가. 별도 venv 전용이며
작성 시점에 실행해 보지 않았다. API 근거: huggingface.co/CohereLabs/cohere-transcribe-03-2026 모델 카드
  AutoProcessor / CohereAsrForConditionalGeneration, processor(audio, sampling_rate=16000, language="ko")
바이어싱 공식 경로가 없으므로 hotwords/context 는 무시하고 meta["biasing"]="unsupported".
"""
from __future__ import annotations

import time

import numpy as np

from .base import OfflineSTT, STTResult


class CohereTransformersSTT(OfflineSTT):
    name = "cohere_transformers"

    def __init__(self, model_dir: str, provider: str = "cpu", language: str = "ko"):
        self.path, self.provider, self.language = model_dir, provider, language or "ko"
        self.model = self.processor = None

    def load(self) -> None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # 지연 import
        self.processor = AutoProcessor.from_pretrained(self.path)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            self.path, device_map="cuda:0" if self.provider == "cuda" else "cpu")

    def transcribe(self, pcm: np.ndarray, sr: int,
                   hotwords: list[str] | None = None, context: str = "") -> STTResult:
        if sr != 16000:
            raise ValueError("cohere_transformers: 16 kHz 입력만 지원")
        t0 = time.perf_counter()
        inputs = self.processor(np.asarray(pcm, np.float32), sampling_rate=sr,
                                return_tensors="pt", language=self.language)
        inputs.to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=256)
        text = self.processor.decode(out, skip_special_tokens=True)
        if isinstance(text, list):
            text = text[0]
        ms = (time.perf_counter() - t0) * 1000
        return STTResult(text.strip(), ms, len(pcm) / sr, meta=dict(
            adapter=self.name, model=self.path, provider=self.provider, biasing="unsupported"))

    def unload(self) -> None:
        self.model = self.processor = None
