"""녹음 전체를 오프라인 어댑터로 인식해 data/results/*.jsonl로 저장하고 채점표를 출력한다.

python -m evaltools.run_eval --config-name A1 [--section stt.final] [--hotwords] [--context] [--limit N]
"""
import argparse
import json
import re
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

from voicecore.adapters import registry
from evaltools import score

ROOT = Path(__file__).resolve().parent.parent
SR = 16000


def get_section(cfg: dict, name: str, a) -> dict:
    """'stt.final' 같은 경로의 섹션을 복사하고 --adapter/--model/--provider를 덮어쓴다."""
    sec = cfg
    for k in name.split("."):
        if k not in sec:
            sys.exit(f"config에 섹션이 없습니다: {name}")
        sec = sec[k]
    sec = dict(sec)
    for k in ("adapter", "model", "provider"):
        if getattr(a, k, None):
            sec[k] = getattr(a, k)
    return sec


def biasing_label(hot: bool, ctx: bool) -> str:
    return "both" if hot and ctx else "hotwords" if hot else "context" if ctx else "none"


def load_manifest(path, limit=None) -> list[dict]:
    rows = score.read_tsv(path)
    if not rows:
        sys.exit(f"manifest가 비어 있습니다: {path}")
    return rows[:limit] if limit else rows


def read_wav(path) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    if not p.is_file():
        sys.exit(f"wav가 없습니다: {path}")
    pcm, sr = sf.read(str(p), dtype="float32", always_2d=True)
    if sr != SR:
        sys.exit(f"{path}: 샘플레이트 {sr} (16000만 지원, 리샘플링 안 함)")
    return pcm.mean(axis=1).astype(np.float32)


def terms_list(path) -> list[str]:
    return list(score.load_terms(path))


def out_path(out_dir, config, adapter, model, biasing) -> Path:
    name = re.sub(r"[^\w.\-]+", "-", f"{config}_{adapter}_{model}_{biasing}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return Path(out_dir) / f"{name}.jsonl"


def common_args(ap):
    ap.add_argument("--config", default=str(ROOT / "config.toml"), help="config.toml 경로")
    ap.add_argument("--manifest", default="data/recordings/manifest.tsv")
    ap.add_argument("--sentences", default="data/sentences/sentences_v1.tsv")
    ap.add_argument("--terms", default="data/vocab/terms_v1.tsv")
    ap.add_argument("--out-dir", default="data/results")
    ap.add_argument("--adapter")
    ap.add_argument("--model")
    ap.add_argument("--provider")
    ap.add_argument("--config-name", default="adhoc", help="A0|A1|B|C 등 구성 이름 (결과 파일명·그룹 키)")
    ap.add_argument("--hotwords", action="store_true", help="terms tsv의 canonical 열을 hotwords로 전달")
    ap.add_argument("--limit", type=int, help="앞에서 N개만")


def main(argv=None):
    ap = argparse.ArgumentParser(description="녹음 전체를 오프라인 어댑터로 인식하고 채점한다.")
    common_args(ap)
    ap.add_argument("--section", default="stt.final")
    ap.add_argument("--context", action="store_true", help="같은 용어 목록을 한 줄 텍스트로 context 전달")
    a = ap.parse_args(argv)

    cfg = tomllib.loads(Path(a.config).read_text(encoding="utf-8"))
    sec = get_section(cfg, a.section, a)
    man = load_manifest(a.manifest, a.limit)
    for p in (a.sentences, a.terms):  # 채점에 필요한 파일을 미리 확인
        if not Path(p).is_file():
            sys.exit(f"파일이 없습니다: {p}")
    terms = terms_list(a.terms) if a.hotwords or a.context else []
    hot = terms if a.hotwords else None
    ctx = ", ".join(terms) if a.context else ""
    bias = biasing_label(a.hotwords, a.context)

    try:
        stt = registry.build_offline(sec, models_dir=cfg.get("models", {}).get("dir", "models"))
        t = time.perf_counter()
        stt.load()
    except NotImplementedError as e:
        sys.exit(f"어댑터를 만들 수 없습니다: {e}")
    load_ms = (time.perf_counter() - t) * 1000  # elapsed_ms에는 넣지 않고 meta에만 기록
    print(f"load {load_ms:.0f} ms", file=sys.stderr)

    path = out_path(a.out_dir, a.config_name, sec["adapter"], sec["model"], bias)
    with open(path, "w", encoding="utf-8") as f:
        for i, m in enumerate(man, 1):
            pcm = read_wav(m["path"])
            res = stt.transcribe(pcm, SR, hotwords=hot, context=ctx)
            f.write(json.dumps({
                "id": m["id"], "ref": m["text"], "hyp": res.text, "elapsed_ms": res.elapsed_ms,
                "audio_sec": len(pcm) / SR, "adapter": sec["adapter"], "model": sec["model"],
                "provider": sec.get("provider", ""), "config": a.config_name, "biasing": bias,
                "meta": {**res.meta, "load_ms": load_ms}}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(man)}] {m['id']} {res.elapsed_ms:.0f} ms", file=sys.stderr)
    stt.unload()
    print(f"-> {path}")
    print(score.run(path, a.sentences, a.terms))


if __name__ == "__main__":
    main()
