"""Engine 테스트(가짜 어댑터): python -m tests.test_engine"""
from __future__ import annotations

import copy
import tempfile
import threading
import time
import tomllib
from pathlib import Path

import numpy as np
import soundfile as sf

from voicecore.adapters import registry
from voicecore.adapters.base import OfflineSTT, StreamingSTT, STTResult
from voicecore.audio import FileSource
from voicecore.engine import Engine
from voicecore.events import EngineError, OutputDone, PartialUpdated, UtteranceFinalized

TARGET = "안녕하세요 반갑습니다 엔진 테스트입니다"
FINAL = "안녕하세요, 반갑습니다. 엔진 테스트입니다."
TMP = Path(tempfile.mkdtemp(prefix="vc-test-"))


class FakeStreaming(StreamingSTT):
    def load(self): pass

    def start(self, hotwords=None):
        self.n = 0

    def feed(self, pcm, sr):           # 청크마다 한 글자씩 늘어나는 partial
        self.n += 1
        return TARGET[:self.n] if self.n <= len(TARGET) else None

    def finish(self):
        return STTResult(TARGET, 1.0)


class FakeOffline(OfflineSTT):
    delay = 0.0

    def load(self): pass

    def transcribe(self, pcm, sr, hotwords=None, context=""):
        time.sleep(self.delay)
        return STTResult(FINAL, self.delay * 1000, len(pcm) / sr)


registry.build_streaming = lambda section, models_dir="models": FakeStreaming()
registry.build_offline = lambda section, models_dir="models": FakeOffline()

with open("config.toml", "rb") as f:
    BASE = tomllib.load(f)
WAV = TMP / "sine1s.wav"
sf.write(WAV, (0.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32), 16000)


def make(second_pass=True, deadline=0, delay=0.0):
    cfg = copy.deepcopy(BASE)
    cfg["output"]["method"] = "none"
    cfg["audio"]["input_device"] = "no-such-device-for-test"   # 마이크 없이(열기 실패 경로)
    cfg["engine"].update(second_pass=second_pass, deadline_verbatim_ms=deadline)
    cfg["vocab"]["file"] = str(TMP / "no_terms.tsv")
    cfg["log"]["dir"] = str(TMP / "log")
    FakeOffline.delay = delay
    evs, done = [], threading.Event()

    def on_event(ev):
        evs.append(ev)
        if isinstance(ev, OutputDone):
            done.set()
    eng = Engine(cfg, on_event)
    eng.load()
    evs.clear()            # 워밍업 이벤트 제외
    return eng, evs, done


def run_one(eng, done, evs):
    src = FileSource(WAV)
    uid = eng.start("verbatim", src)
    t = time.time()                # 파일 끝 + 1차 backlog 소진(마지막 partial)까지 기다린 뒤 stop
    while not (src.eof and any(isinstance(e, PartialUpdated) and e.stable + e.unstable == TARGET
                               for e in evs)) and time.time() - t < 5:
        time.sleep(0.005)
    eng.stop()
    eng.stop()             # 중복 stop 안전
    assert done.wait(5), "OutputDone 없음"
    time.sleep(0.1)        # 늦은 이벤트 확인용
    return uid


def kinds(evs):
    return [type(e).__name__ for e in evs]


def test_flow(second_pass):
    eng, evs, done = make(second_pass)
    uid = run_one(eng, done, evs)
    k = kinds(evs)
    assert k.count("PartialUpdated") >= 1 and k.count("OutputDone") == 1, k
    assert "EngineError" not in k, [e for e in evs if isinstance(e, EngineError)]
    fi, od = k.index("UtteranceFinalized"), k.index("OutputDone")
    assert max(i for i, x in enumerate(k) if x == "PartialUpdated") < fi < od == len(k) - 1, k
    fin = evs[fi]
    assert fin.utt_id == uid and all(e.utt_id == uid for e in evs)
    want = FINAL if second_pass else TARGET
    assert fin.raw == want and fin.text == want, (fin.raw, fin.text)
    assert {"stt_final", "correction", "llm", "output", "total"} <= set(fin.timings_ms), fin.timings_ms
    assert evs[od].ok is True and evs[od].method == "none"
    parts = [e for e in evs if isinstance(e, PartialUpdated)]
    assert all(TARGET.startswith(p.stable + p.unstable) for p in parts)
    assert parts[-1].stable and len(parts[-1].stable) <= len(parts[-1].stable + parts[-1].unstable)
    eng.close()
    print(f"flow second_pass={second_pass}: OK  partials={len(parts)} timings={fin.timings_ms}")


def test_deadline():
    eng, evs, done = make(True, deadline=1, delay=0.02)
    run_one(eng, done, evs)
    k = kinds(evs)
    assert "EngineError" in k and "UtteranceFinalized" not in k, k
    assert not any(isinstance(e, OutputDone) and e.ok for e in evs), k
    eng.close()
    print("deadline 1ms: OK  ", [e.message for e in evs if isinstance(e, EngineError)])


def test_cancel():
    eng, evs, done = make(True)
    src = FileSource(WAV, realtime=True)
    eng.start("verbatim", src)
    time.sleep(0.2)
    eng.cancel()
    eng.stop()             # 취소 뒤 stop 은 무시
    time.sleep(0.5)
    k = kinds(evs)
    assert "OutputDone" not in k and "UtteranceFinalized" not in k, k
    eng.close()
    print("cancel: OK")


def test_overlap():
    # 처리 중에 다음 start 가 와도 각 발화가 자기 출력을 정확히 1회
    eng, evs, done = make(True, delay=0.2)
    ids = []
    for _ in range(2):
        src = FileSource(WAV)
        ids.append(eng.start("verbatim", src))
        while not src.eof:
            time.sleep(0.005)
        eng.stop()
    t = time.time()
    while sum(isinstance(e, OutputDone) for e in evs) < 2 and time.time() - t < 5:
        time.sleep(0.01)
    assert ids[0] != ids[1]
    for uid in ids:
        od = [e for e in evs if isinstance(e, OutputDone) and e.utt_id == uid]
        fin = [e for e in evs if isinstance(e, UtteranceFinalized) and e.utt_id == uid]
        assert len(od) == 1 and od[0].ok and len(fin) == 1 and fin[0].raw == FINAL, (uid, kinds(evs))
    eng.close()
    print("overlap start: OK", ids)


def test_callback_error():
    eng, evs, done = make(False)
    orig = eng.on_event

    def bad(ev):
        orig(ev)
        if isinstance(ev, PartialUpdated):
            raise RuntimeError("boom")
    eng.on_event = bad
    run_one(eng, done, evs)
    k = kinds(evs)
    assert "EngineError" in k and k.count("OutputDone") == 1, k
    eng.close()
    print("callback exception -> EngineError: OK")


def main():
    test_flow(True)
    test_flow(False)
    test_deadline()
    test_cancel()
    test_overlap()
    test_callback_error()


if __name__ == "__main__":
    main()
