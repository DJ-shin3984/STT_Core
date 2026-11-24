"""스트리밍 어댑터 평가: 첫 Partial 지연, 종료 flush 지연. 결과는 run_eval과 같은 JSONL 스키마.

python -m evaltools.stream_eval --config-name A0 [--realtime] [--limit N]
첫 Partial 지연 = 첫 비어있지 않은 partial 수신 시각 - 음성 시작 시각(RMS 임계값을 넘은 첫 청크의 시작).
 --realtime: 실제 시간에 맞춰 feed (wall clock 그대로).
 기본(빠른 모드): 가상 시계 = 청크 끝 오디오 시각 + 그 feed 호출 소요 시간.
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from voicecore.adapters import registry
from evaltools import score
from evaltools.latency import Tracer
from evaltools.run_eval import SR, biasing_label, common_args, get_section, load_manifest, out_path, read_wav, terms_list


def run_one(stt, pcm, chunk, realtime, rms_thr, hot):
    cs = chunk / SR
    stt.start(hotwords=hot)
    t0 = time.perf_counter()
    speech_at = first = None
    for j, i in enumerate(range(0, len(pcm), chunk)):
        c = pcm[i:i + chunk]
        if speech_at is None and np.sqrt(np.mean(c ** 2)) > rms_thr:
            speech_at = j * cs
        if realtime:
            time.sleep(max(0.0, t0 + (j + 1) * cs - time.perf_counter()))
        a = time.perf_counter()
        p = stt.feed(c, SR)
        b = time.perf_counter()
        if first is None and speech_at is not None and p and p.strip():
            first = ((b - t0) if realtime else (j + 1) * cs + (b - a)) - speech_at
    tr = Tracer()
    tr.mark("fed")  # 마지막 청크 feed가 끝난 시점
    res = stt.finish()
    tr.mark("done")
    return res, tr.elapsed_ms("fed", "done"), None if first is None else first * 1000


def main(argv=None):
    ap = argparse.ArgumentParser(description="스트리밍 어댑터의 첫 Partial 지연·종료 flush 지연을 측정한다.")
    common_args(ap)
    ap.add_argument("--section", default="stt.streaming")
    ap.add_argument("--realtime", action="store_true", help="실제 시간에 맞춰 sleep하며 feed")
    ap.add_argument("--rms", type=float, default=0.01, help="음성 시작 판정 RMS 임계값")
    a = ap.parse_args(argv)

    cfg = tomllib.loads(Path(a.config).read_text(encoding="utf-8"))
    sec = get_section(cfg, a.section, a)
    man = load_manifest(a.manifest, a.limit)
    for p in (a.sentences, a.terms):
        if not Path(p).is_file():
            sys.exit(f"파일이 없습니다: {p}")
    chunk = int(SR * cfg["audio"]["chunk_ms"] / 1000)
    hot = terms_list(a.terms) if a.hotwords else None
    bias = biasing_label(a.hotwords, False)

    try:
        stt = registry.build_streaming(sec, models_dir=cfg.get("models", {}).get("dir", "models"))
        t = time.perf_counter()
        stt.load()
    except NotImplementedError as e:
        sys.exit(f"어댑터를 만들 수 없습니다: {e}")
    load_ms = (time.perf_counter() - t) * 1000

    path = out_path(a.out_dir, a.config_name, sec["adapter"], sec["model"], bias)
    firsts = []
    with open(path, "w", encoding="utf-8") as f:
        for i, m in enumerate(man, 1):
            pcm = read_wav(m["path"])
            res, flush_ms, first_ms = run_one(stt, pcm, chunk, a.realtime, a.rms, hot)
            if first_ms is not None:
                firsts.append(first_ms)
            f.write(json.dumps({
                "id": m["id"], "ref": m["text"], "hyp": res.text, "elapsed_ms": flush_ms,
                "audio_sec": len(pcm) / SR, "adapter": sec["adapter"], "model": sec["model"],
                "provider": sec.get("provider", ""), "config": a.config_name, "biasing": bias,
                "meta": {**res.meta, "first_partial_ms": first_ms, "load_ms": load_ms,
                         "realtime": a.realtime, "chunk_ms": cfg["audio"]["chunk_ms"]}}, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(man)}] {m['id']} first={first_ms and round(first_ms)} flush={flush_ms:.0f} ms", file=sys.stderr)
    stt.unload()
    fl = [r["elapsed_ms"] for r in score.load_jsonl(path)]
    p = lambda v, q: "-" if not v else f"{score.pct(v, q):.0f}"
    print(f"-> {path}  (realtime={a.realtime}, partial 없음 {len(man) - len(firsts)}건)")
    print(f"첫 Partial 지연 ms: p50={p(firsts, 50)} p95={p(firsts, 95)} | 종료 flush 지연 ms: p50={p(fl, 50)} p95={p(fl, 95)}")
    print(score.run(path, a.sentences, a.terms))


if __name__ == "__main__":
    main()
