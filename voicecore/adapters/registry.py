"""Build adapters from a config.toml section.

    cfg = tomllib.load(open("config.toml", "rb"))
    final = build_offline(cfg["stt"]["final"], models_dir=cfg["models"]["dir"])
    live  = build_streaming(cfg["stt"]["streaming"], models_dir=cfg["models"]["dir"])

load() 는 호출자가 한다(여기서는 객체만 만든다).
"""
from __future__ import annotations

from pathlib import Path

from .base import OfflineSTT, StreamingSTT


def _model_dir(section: dict, models_dir: str) -> str:
    p = Path(models_dir) / section["model"]
    if not p.exists():
        raise FileNotFoundError(
            f"모델 없음: {p}\n  python -m voicecore.adapters.download {section['model']}")
    return str(p)


def build_offline(section: dict, models_dir: str = "models") -> OfflineSTT:
    kind = section["adapter"]
    kw = dict(provider=section.get("provider", "cpu"), language=section.get("language", "ko"))
    if kind == "sherpa_offline_qwen3asr":
        from .sherpa_offline_qwen3asr import SherpaQwen3ASRSTT as cls
    elif kind == "sherpa_offline_cohere":
        from .sherpa_offline_cohere import SherpaCohereSTT as cls
    elif kind == "qwen3asr_transformers":
        from .qwen3asr_transformers import Qwen3ASRTransformersSTT as cls
    elif kind == "cohere_transformers":
        from .cohere_transformers import CohereTransformersSTT as cls
    else:
        raise ValueError(f"알 수 없는 offline adapter: {kind}")
    return cls(_model_dir(section, models_dir), **kw)


def build_streaming(section: dict, models_dir: str = "models") -> StreamingSTT:
    kind = section["adapter"]
    if kind != "sherpa_online":
        raise ValueError(f"알 수 없는 streaming adapter: {kind}")
    from .sherpa_online import SherpaOnlineSTT
    return SherpaOnlineSTT(
        _model_dir(section, models_dir), provider=section.get("provider", "cpu"),
        language=section.get("language", "ko"), decoding=section.get("decoding", "greedy_search"),
        hotwords_score=float(section.get("hotwords_score", 2.0)))
