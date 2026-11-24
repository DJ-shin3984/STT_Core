"""python -m tests.test_run_eval -- 가짜 어댑터로 run_eval / stream_eval / score 연결 확인."""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from evaltools import run_eval, stream_eval
from voicecore.adapters import registry
from voicecore.adapters.base import OfflineSTT, StreamingSTT, STTResult

seen = {}


class FakeOffline(OfflineSTT):
    def load(self):
        pass

    def transcribe(self, pcm, sr, hotwords=None, context=""):
        seen.update(hotwords=hotwords, context=context, sr=sr)
        return STTResult("MariaDB 접속", 123.0, len(pcm) / sr, {"biasing": "none"})


class FakeStreaming(StreamingSTT):
    def load(self):
        pass

    def start(self, hotwords=None):
        self.n = 0

    def feed(self, pcm, sr):
        self.n += 1
        return "마리" if self.n == 30 else None  # 30번째 청크(0.6s)에서 첫 partial

    def finish(self):
        return STTResult("MariaDB 접속", 5.0)


def tone(path, sec, sr=16000, silence=0.0):
    t = np.arange(int(sr * sec)) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    x[:int(sr * silence)] = 0
    sf.write(path, x, sr)


def run(mod, args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        mod.main(args)
    return buf.getvalue()


def main():
    registry.build_offline = lambda section, models_dir="models": FakeOffline()
    registry.build_streaming = lambda section, models_dir="models": FakeStreaming()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tone(td / "a.wav", 1.0)
        tone(td / "b.wav", 4.0, silence=0.2)
        (td / "manifest.tsv").write_text(
            "id\ttext\tpath\tduration_sec\trecorded_at\tcategory\tspeed\n"
            f"d1\tMariaDB 접속\t{td / 'a.wav'}\t1.0\t2026-10-08\tdev\tnormal\n"
            f"t1\t마리아 접속\t{td / 'b.wav'}\t4.0\t2026-10-08\ttrap\tnormal\n", encoding="utf-8")
        (td / "s.tsv").write_text(
            "id\tcategory\tspeed\tsplit\ttext\tterms\nd1\tdev\tnormal\ttune\tMariaDB 접속\tMariaDB\n"
            "t1\ttrap\tnormal\tholdout\t마리아 접속\t\n", encoding="utf-8")
        (td / "t.tsv").write_text("canonical\taliases\nMariaDB\t마리아디비\n", encoding="utf-8")
        base = ["--manifest", str(td / "manifest.tsv"), "--sentences", str(td / "s.tsv"),
                "--terms", str(td / "t.tsv"), "--out-dir", str(td / "res")]

        out = run(run_eval, base + ["--config-name", "A1", "--hotwords"])
        f = td / "res" / "A1_sherpa_offline_qwen3asr_qwen3-asr-1.7b-int8_hotwords.jsonl"
        assert f.is_file(), list((td / "res").iterdir())
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        assert [r["id"] for r in rows] == ["d1", "t1"] and rows[0]["ref"] == "MariaDB 접속"
        assert rows[0]["hyp"] == "MariaDB 접속" and rows[0]["elapsed_ms"] == 123.0  # load 시간 미포함
        assert rows[1]["audio_sec"] == 4.0 and rows[0]["biasing"] == "hotwords" and rows[0]["config"] == "A1"
        assert "load_ms" in rows[0]["meta"] and seen["hotwords"] == ["MariaDB"] and seen["context"] == "" and seen["sr"] == 16000
        assert "| A1 |" in out and "1/1" in out and "1/1 (100.0%)" in out, out  # 함정 오보정 1/1, 재현율 1/1
        assert len(run_eval.load_manifest(td / "manifest.tsv", 1)) == 1

        run(run_eval, base + ["--config-name", "A1", "--context", "--limit", "1"])
        assert seen["context"] == "MariaDB" and seen["hotwords"] is None

        # 스트리밍: 음성 시작 = 0s(a.wav) 또는 0.2s(b.wav 앞 0.2s 무음), 첫 partial = 0.6s 청크 끝
        out = run(stream_eval, base + ["--config-name", "A0"])
        g = td / "res" / "A0_sherpa_online_nemotron-3.5-asr-streaming-0.6b-160ms-int8_none.jsonl"
        rows = [json.loads(l) for l in g.read_text(encoding="utf-8").splitlines()]
        assert abs(rows[0]["meta"]["first_partial_ms"] - 600) < 50, rows[0]["meta"]
        assert abs(rows[1]["meta"]["first_partial_ms"] - 400) < 50, rows[1]["meta"]
        assert 0 <= rows[0]["elapsed_ms"] < 100 and "첫 Partial 지연" in out

        # 16kHz가 아니면 에러
        tone(td / "bad.wav", 1.0, sr=44100)
        try:
            run_eval.read_wav(td / "bad.wav")
            raise AssertionError("SystemExit expected")
        except SystemExit as e:
            assert "44100" in str(e)
        # 파일 없음 -> 명확한 에러
        try:
            run(run_eval, ["--manifest", str(td / "nope.tsv")])
            raise AssertionError("SystemExit expected")
        except SystemExit as e:
            assert "nope.tsv" in str(e)
    print("test_run_eval OK")


if __name__ == "__main__":
    main()
