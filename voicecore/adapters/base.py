"""STT adapter interface. All adapters take float32 mono PCM in [-1, 1].

Two kinds:
  OfflineSTT   - whole utterance in, text out (2nd pass / evaluation)
  StreamingSTT - feed chunks, read partials, finish() flushes (1st pass)
Vocabulary biasing goes in as `hotwords` (list of canonical terms) and/or
`context` (free text prompt). Adapters that support neither must ignore both
and set result.meta["biasing"] = "unsupported".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class STTResult:
    text: str
    elapsed_ms: float                 # wall time of the call(s) that produced `text`
    audio_sec: float = 0.0
    meta: dict = field(default_factory=dict)   # adapter name, model, provider, biasing, ...


class OfflineSTT(ABC):
    name: str = "offline"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, pcm: np.ndarray, sr: int,
                   hotwords: list[str] | None = None, context: str = "") -> STTResult: ...

    def unload(self) -> None:  # optional
        pass


class StreamingSTT(ABC):
    name: str = "streaming"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def start(self, hotwords: list[str] | None = None) -> None:
        """Begin a new utterance."""

    @abstractmethod
    def feed(self, pcm: np.ndarray, sr: int) -> str | None:
        """Push one chunk. Returns the current partial text if it changed, else None."""

    @abstractmethod
    def finish(self) -> STTResult:
        """Signal end of utterance, flush, return final text."""

    def unload(self) -> None:  # optional
        pass
