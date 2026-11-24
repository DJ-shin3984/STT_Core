"""Core -> Shell events. Core never returns text; it publishes these."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PartialUpdated:        # while speaking (overlay only)
    utt_id: str
    stable: str              # front part unchanged for N updates (fast-corrected)
    unstable: str            # changing tail (never corrected)


@dataclass
class SegmentStable:         # a new chunk moved into the stable zone
    utt_id: str
    text: str


@dataclass
class UtteranceFinalized:    # after PTT release: final text decided
    utt_id: str
    mode: str                # "verbatim" | "polished"
    raw: str                 # 2nd-pass STT text (or 1st-pass flush if no 2nd pass)
    text: str                # after fast correction (+ LLM when polished) + safety check
    timings_ms: dict = field(default_factory=dict)   # stt_final, correction, llm, output, total


@dataclass
class OutputDone:
    utt_id: str
    ok: bool
    method: str              # "clipboard" | "none"


@dataclass
class EngineError:
    utt_id: str
    message: str
