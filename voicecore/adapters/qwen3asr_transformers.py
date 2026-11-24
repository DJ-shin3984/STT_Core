"""Qwen3-ASR transformers 백엔드 (qwen-asr 패키지). 1-0 비교용.

**미검증**: 이 프로젝트 기본 venv 에는 torch/transformers/qwen-asr 가 없다(의존성 충돌).
별도 venv(qwen-asr 0.0.6 + transformers==4.57.6)에서만 실행 가능하며, 작성 시점에 실행해 보지 않았다.
API 근거: QwenLM/Qwen3-ASR README, qwen_asr/inference/qwen3_asr.py
  Qwen3ASRModel.from_pretrained(path, dtype=..., device_map=..., max_new_tokens=...)
  transcribe(audio=(np.ndarray, sr), context=str, language="Korean"|None) -> [result(.text, .language)]
hotwords 는 별도 인자가 없으므로 context 끝에 붙인다.
"""
from __future__ import annotations

import time

import numpy as np

from ._lang import lang_name
from .base import OfflineSTT, STTResult


class Qwen3ASRTransformersSTT(OfflineSTT):
    name = "qwen3asr_transformers"

    def __init__(self, model_dir: str, provider: str = "cpu", language: str = "ko"):
        self.path, self.provider, self.language = model_dir, provider, lang_name(language)
        self.model = None

    def load(self) -> None:
        import torch  # 지연 import
        from qwen_asr import Qwen3ASRModel
        cuda = self.provider == "cuda"
        self.model = Qwen3ASRModel.from_pretrained(
            self.path, dtype=torch.bfloat16 if cuda else torch.float32,
            device_map="cuda:0" if cuda else "cpu", max_inference_batch_size=1, max_new_tokens=256)

    def transcribe(self, pcm: np.ndarray, sr: int,
                   hotwords: list[str] | None = None, context: str = "") -> STTResult:
        ctx = context
        if hotwords:  # 용어 목록을 context 로 전달
            ctx = (ctx + "\n" if ctx else "") + ", ".join(hotwords)
        t0 = time.perf_counter()
        r = self.model.transcribe(audio=(np.asarray(pcm, np.float32), sr),
                                  context=ctx, language=self.language)[0]
        ms = (time.perf_counter() - t0) * 1000
        return STTResult(r.text, ms, len(pcm) / sr, meta=dict(
            adapter=self.name, model=self.path, provider=self.provider,
            biasing="context" if ctx else "off", language=r.language))

    def unload(self) -> None:
        self.model = None
