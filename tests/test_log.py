"""CorrectionLog 테스트: python -m tests.test_log"""
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voicecore.log import CorrectionLog

SR = 16000


def pcm(sec=2.0):
    t = np.arange(int(SR * sec)) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def lines(d):
    return [json.loads(l) for l in (Path(d) / "log.jsonl").read_text(encoding="utf-8").splitlines()]


def main():
    tmp = tempfile.mkdtemp()
    try:
        # 1) 상한 넉넉: wav 3개, jsonl 3줄, mark_user_fix 1줄
        log = CorrectionLog(tmp, max_mb=100)
        for i in range(3):
            p = log.write(f"u{i}", pcm(), SR, {"mode": "verbatim", "raw": f"원문{i}", "final": f"최종{i}"})
            assert p == Path(tmp) / f"u{i}.wav"
            log.flush()
            time.sleep(0.02)
        assert sorted(x.name for x in Path(tmp).glob("*.wav")) == ["u0.wav", "u1.wav", "u2.wav"]
        data, sr = sf.read(Path(tmp) / "u0.wav", dtype="int16")
        assert sr == SR and len(data) == SR * 2
        rows = lines(tmp)
        assert len(rows) == 3 and rows[1]["utt_id"] == "u1" and rows[1]["raw"] == "원문1" and "ts" in rows[1]
        log.mark_user_fix("u1", "사용자가 고친 문장")
        log.flush()
        rows = lines(tmp)
        assert len(rows) == 4 and rows[3] == {"ts": rows[3]["ts"], "utt_id": "u1", "user_fix": "사용자가 고친 문장"}
        print("OK wav 3개 / jsonl 3줄 / user_fix 1줄")

        # 2) 상한 초과: 오래된 wav부터 삭제, jsonl은 유지 (wav 1개 약 64KB, 상한 0.15MB)
        log.max_bytes = int(0.15 * 1024 * 1024)
        log.write("u3", pcm(), SR, {"mode": "polished"})
        log.flush()
        wavs = sorted(x.name for x in Path(tmp).glob("*.wav"))
        assert "u0.wav" not in wavs and "u3.wav" in wavs, wavs
        assert len(lines(tmp)) == 5
        print("OK 오래된 wav 삭제, jsonl 유지:", wavs)
        log.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_log 통과")


if __name__ == "__main__":
    main()
