"""모델 다운로드: k2-fsa GitHub 릴리스 tar.bz2 → models/<name>/

    python -m voicecore.adapters.download --list
    python -m voicecore.adapters.download zipformer-korean-2024-06-16
    python -m voicecore.adapters.download --all

카탈로그 URL은 모두 HTTP HEAD로 존재를 확인한 것(2026-10-08).
qwen3-asr-1.7b-int8 은 GitHub 릴리스에 없음(modelscope 전용) → 카탈로그 미포함.
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
CATALOG = {  # 이름(config.toml의 model 값) → tarball 파일명
    "nemotron-3.5-asr-streaming-0.6b-160ms-int8": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2",
    "nemotron-3.5-asr-streaming-0.6b-560ms-int8": "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
    "zipformer-korean-2024-06-16": "sherpa-onnx-streaming-zipformer-korean-2024-06-16.tar.bz2",
    "qwen3-asr-0.6b-int8": "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
    "cohere-transcribe-14-lang-int8": "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2",
    "sense-voice-zh-en-ja-ko-yue-2024-07-17-int8": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
}

# GitHub 릴리스에 없는 모델 → 수동 배치 안내(int8: conv_frontend + encoder.int8 + decoder.int8 ≈ 2.4GB, tokenizer 는 0.6B 와 동일)
MANUAL = {"qwen3-asr-1.7b-int8": "https://modelscope.cn/models/zengshuishui/Qwen3-ASR-onnx/tree/master/model_1.7B"}


def download(name: str, models_dir: str = "models") -> Path:
    dest = Path(models_dir) / name
    if dest.exists():
        print(f"[skip] {dest} 이미 있음")
        return dest
    url = _BASE + CATALOG[name]
    tmp = Path(models_dir) / f".tmp-{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    tarball = tmp / CATALOG[name]
    print(f"[get] {url}")
    with urllib.request.urlopen(url) as r, open(tarball, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done >> 20}/{total >> 20} MB", end="", flush=True)
    print("\n[untar]")
    with tarfile.open(tarball, "r:bz2") as t:
        t.extractall(tmp, filter="data")
    tarball.unlink()  # 압축 해제 후 삭제
    (top,) = [p for p in tmp.iterdir() if p.is_dir()]  # 최상위 폴더 1개
    top.rename(dest)
    tmp.rmdir()
    print(f"[ok] {dest}")
    return dest


def main(argv: list[str]) -> None:
    if not argv or argv[0] == "--list":
        for k, v in CATALOG.items():
            print(f"{k:48s} {_BASE}{v}")
        return
    for name in (CATALOG if argv[0] == "--all" else argv):
        if name in MANUAL:
            sys.exit(f"{name}: GitHub 릴리스에 없음. {MANUAL[name]} 에서 받아 models/{name}/ 에 배치")
        if name not in CATALOG:
            sys.exit(f"알 수 없는 모델: {name} (--list 참고)")
        download(name)


if __name__ == "__main__":
    main(sys.argv[1:])
