"""CLI: 단축키 없이 엔진 전체 경로 실행.

    python -m voicecore --wav path [--mode verbatim|polished] [--no-paste] [--realtime] [--no-second-pass]
    python -m voicecore --mic --seconds N [--no-paste]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import tomllib

from .audio import FileSource
from .engine import Engine
from .events import EngineError, OutputDone


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voicecore")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav")
    src.add_argument("--mic", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--mode", choices=["verbatim", "polished"], default="verbatim")
    ap.add_argument("--no-paste", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--no-second-pass", action="store_true")
    ap.add_argument("--config", default="config.toml")
    a = ap.parse_args(argv)

    with open(a.config, "rb") as f:
        cfg = tomllib.load(f)
    if a.no_paste:
        cfg["output"]["paste"] = False
    if a.no_second_pass:
        cfg["engine"]["second_pass"] = False

    t0, done, errors = [time.perf_counter()], threading.Event(), []

    def on_event(ev):
        print(f"[{(time.perf_counter() - t0[0]) * 1000:7.0f} ms] {ev}", flush=True)
        if isinstance(ev, EngineError):
            errors.append(ev)
        if isinstance(ev, OutputDone):
            done.set()

    eng = Engine(cfg, on_event)
    eng.load()
    try:
        source = FileSource(a.wav, cfg["audio"]["chunk_ms"], a.realtime, cfg["audio"]["sample_rate"]) \
            if a.wav else None
        t0[0] = time.perf_counter()
        print(f"[      0 ms] start mode={a.mode} second_pass={cfg['engine']['second_pass']}", flush=True)
        eng.start(a.mode, source)
        if source is not None:
            while not source.eof:
                time.sleep(0.01)
        else:
            time.sleep(a.seconds)
        print(f"[{(time.perf_counter() - t0[0]) * 1000:7.0f} ms] stop", flush=True)
        eng.stop()
        done.wait()
    finally:
        eng.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
