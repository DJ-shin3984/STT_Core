"""수정 로그. API (고정):
    log = CorrectionLog(dir="data/log", max_mb=2048)
    path = log.write(utt_id, pcm, sr, record)   # <utt_id>.wav(PCM16) 저장 + log.jsonl에 한 줄 append (큐 + 스레드 1개)
    log.mark_user_fix(utt_id, fixed_text)       # "직전 입력 수정" 결과 = 정답. {"ts","utt_id","user_fix"} 한 줄
    log.flush(); log.close()
디스크 상한: write 후 폴더 합계가 max_mb를 넘으면 오래된 wav부터 삭제(jsonl은 절대 안 지움, 방금 쓴 wav는 남김).
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import soundfile as sf


class CorrectionLog:
    def __init__(self, dir: str = "data/log", max_mb: float = 2048):
        self.dir, self.max_bytes = Path(dir), int(max_mb * 1024 * 1024)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="correction-log", daemon=True).start()

    def write(self, utt_id: str, pcm, sr: int, record: dict) -> Path:
        path = self.dir / f"{utt_id}.wav"
        self._q.put(("wav", path, pcm, sr, {"ts": time.time(), "utt_id": utt_id, **record}))
        return path

    def mark_user_fix(self, utt_id: str, fixed_text: str):
        self._q.put(("line", {"ts": time.time(), "utt_id": utt_id, "user_fix": fixed_text}))

    def flush(self):
        self._q.join()

    def close(self):
        self.flush()
        self._q.put(None)

    def _run(self):
        while (job := self._q.get()) is not None:
            try:
                if job[0] == "wav":
                    _, path, pcm, sr, rec = job
                    sf.write(str(path), pcm, sr, subtype="PCM_16")
                    self._append(rec)
                    self._trim(path)
                else:
                    self._append(job[1])
            except Exception as e:   # 로그 실패가 입력 흐름을 막으면 안 된다
                print(f"[log] 기록 실패: {e}")
            finally:
                self._q.task_done()
        self._q.task_done()

    def _append(self, rec: dict):
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _trim(self, keep: Path):
        files = [p for p in self.dir.iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        for p in sorted((p for p in files if p.suffix == ".wav" and p != keep), key=lambda p: p.stat().st_mtime):
            if total <= self.max_bytes:
                break
            total -= p.stat().st_size
            p.unlink()
