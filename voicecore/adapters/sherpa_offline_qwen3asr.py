"""sherpa-onnx Qwen3-ASR 오프라인 어댑터 (OfflineRecognizer.from_qwen3_asr).

- hotwords: 스트림 옵션 "hotwords"(쉼표 구분, #3468/#3477) → 호출마다 바꿀 수 있다.
- 언어 힌트: 스트림 옵션 "language"="Korean" 등(#3472). "" / "auto" 면 자동 감지.
- context(자유 텍스트 프롬프트): 이 경로에는 입력이 없어 무시하고 meta["context"]="ignored".
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sherpa_onnx

from ._lang import lang_name
from .base import OfflineSTT, STTResult


def _pick(d: Path, stem: str) -> str:
    c = sorted(d.glob(f"{stem}.int8.onnx")) or sorted(d.glob(f"{stem}.onnx"))
    if not c:
        raise FileNotFoundError(f"{d}/{stem}*.onnx 없음")
    return str(c[0])


class SherpaQwen3ASRSTT(OfflineSTT):
    name = "sherpa_offline_qwen3asr"

    def __init__(self, model_dir: str, provider: str = "cpu", language: str = "ko",
                 num_threads: int = 4, max_new_tokens: int = 256):
        self.dir, self.provider, self.language = Path(model_dir), provider, lang_name(language)
        self.num_threads, self.max_new_tokens = num_threads, max_new_tokens
        self.rec = None

    def load(self) -> None:
        d = self.dir
        self.rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=_pick(d, "conv_frontend"), encoder=_pick(d, "encoder"),
            decoder=_pick(d, "decoder"), tokenizer=str(d / "tokenizer"),
            num_threads=self.num_threads, provider=self.provider,
            max_new_tokens=self.max_new_tokens, max_total_len=1024)

    def transcribe(self, pcm: np.ndarray, sr: int,
                   hotwords: list[str] | None = None, context: str = "") -> STTResult:
        t0 = time.perf_counter()
        s = self.rec.create_stream()
        if self.language:
            s.set_option("language", self.language)
        if hotwords:
            s.set_option("hotwords", ",".join(h.replace(",", " ") for h in hotwords))
        s.accept_waveform(sr, np.asarray(pcm, dtype=np.float32))
        self.rec.decode_stream(s)
        ms = (time.perf_counter() - t0) * 1000
        meta = dict(adapter=self.name, model=self.dir.name, provider=self.provider,
                    biasing="hotwords" if hotwords else "off", language=self.language or "auto")
        if context:
            meta["context"] = "ignored"
        return STTResult(s.result.text.strip(), ms, len(pcm) / sr, meta=meta)

    def unload(self) -> None:
        self.rec = None
