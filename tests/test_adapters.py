"""어댑터 스모크 테스트: python -m tests.test_adapters

models/ 에 있는 sherpa 모델만 검사(없으면 SKIP). 각 모델의 test_wavs 중 짧은 2개 +
zipformer-ko 의 한국어 wav 1개로 hotwords 없음/있음 두 번 실행한다.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicecore.adapters.registry import build_offline, build_streaming

MODELS = Path("models")
KO_WAV = MODELS / "zipformer-korean-2024-06-16" / "test_wavs" / "3.wav"
HOTWORDS = ["주민등록증", "지하철"]
STREAMING = ["zipformer-korean-2024-06-16", "nemotron-3.5-asr-streaming-0.6b-160ms-int8",
             "nemotron-3.5-asr-streaming-0.6b-560ms-int8"]
OFFLINE = ["qwen3-asr-0.6b-int8", "qwen3-asr-1.7b-int8", "cohere-transcribe-14-lang-int8"]


def _dur(w: Path) -> float:
    try:
        return sf.info(w).duration
    except sf.LibsndfileError:  # 일부 모델 동봉 wav(예: cohere ja.wav)는 libsndfile 이 못 읽음
        return 1e9


def wavs(name: str) -> list[Path]:
    short = [w for w in sorted((MODELS / name / "test_wavs").glob("*.wav")) if _dur(w) <= 12]
    ko = [w for w in short if w.stem.startswith("ko")]
    return (ko or short[:2]) + ([KO_WAV] if KO_WAV.exists() else [])


def read(w: Path) -> tuple[np.ndarray, int]:
    pcm, sr = sf.read(w, dtype="float32")
    return (pcm[:, 0] if pcm.ndim > 1 else pcm), sr


def test_streaming(name: str) -> None:
    a = build_streaming({"adapter": "sherpa_online", "model": name, "language": "ko"})
    a.load()
    for hw in (None, HOTWORDS):
        for w in wavs(name):
            pcm, sr = read(w)
            a.start(hw)
            n, parts, cost = sr // 50, 0, 0.0
            for i in range(0, len(pcm), n):  # 20ms 청크
                t = time.perf_counter()
                parts += a.feed(pcm[i:i + n], sr) is not None
                cost += time.perf_counter() - t
            r = a.finish()
            assert parts >= 1, f"{name} {w.name}: partial 없음"
            assert r.text, f"{name} {w.name}: 빈 결과"
            print(f"  [{'hw' if hw else '--'}] {w.name:10s} partials={parts:2d} "
                  f"feed/20ms={cost * 1000 / (len(pcm) / n):5.1f}ms flush={r.elapsed_ms:5.0f}ms "
                  f"biasing={r.meta['biasing']} | {r.text}")
    a.unload()


def test_offline(name: str) -> None:
    kind = "sherpa_offline_cohere" if name.startswith("cohere") else "sherpa_offline_qwen3asr"
    a = build_offline({"adapter": kind, "model": name, "language": "ko"})
    a.load()
    for hw in (None, HOTWORDS):
        for w in wavs(name):
            pcm, sr = read(w)
            r = a.transcribe(pcm, sr, hotwords=hw, context="회의록")
            assert r.text, f"{name} {w.name}: 빈 결과"
            print(f"  [{'hw' if hw else '--'}] {w.name:10s} {r.audio_sec:5.2f}s elapsed={r.elapsed_ms:5.0f}ms "
                  f"biasing={r.meta['biasing']} | {r.text}")
    a.unload()


def test_missing_model() -> None:
    try:
        build_offline({"adapter": "sherpa_offline_qwen3asr", "model": "no-such-model"})
    except FileNotFoundError as e:
        assert "voicecore.adapters.download no-such-model" in str(e)
    else:
        raise AssertionError("FileNotFoundError 기대")
    print("missing-model: OK")


def main() -> None:
    test_missing_model()
    for names, fn in ((STREAMING, test_streaming), (OFFLINE, test_offline)):
        for name in names:
            if not (MODELS / name).exists():
                print(f"SKIP {name} (python -m voicecore.adapters.download {name})")
                continue
            print(f"{fn.__name__}: {name}")
            fn(name)
    print("ALL OK")


if __name__ == "__main__":
    main()
