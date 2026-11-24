"""섹션별 어댑터를 순서대로 적재하며 프로세스 RSS와 GPU 메모리(nvidia-smi)를 기록한다.

python -m evaltools.vram --sections stt.streaming stt.final llm [--wav x.wav]
앞 섹션은 적재된 채로 두므로 마지막 행이 동시 적재 상태다. peak는 적재+인식(--wav) 동안 50ms 간격 폴링한 최댓값.
"""
import argparse
import ctypes
import shutil
import subprocess
import sys
import threading
import tomllib
from ctypes import wintypes
from pathlib import Path

from voicecore.adapters import registry
from evaltools.run_eval import SR, ROOT, get_section, read_wav


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QPP", "QP", "QPNP", "QNP", "Pagefile", "PeakPagefile")]


def rss_mb() -> float | None:
    """현재 프로세스 working set (MiB)."""
    if sys.platform == "win32":
        k32, ps = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        m = _PMC()
        m.cb = ctypes.sizeof(m)
        return m.WorkingSetSize / 2**20 if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb) else None
    try:  # 리눅스 대비
        return int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1024
    except OSError:
        return None


def gpu_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return float(out[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def snap() -> tuple:
    return rss_mb(), gpu_mb()


class Poll:
    """with 블록 동안 50ms 간격으로 (rss, gpu) 최댓값을 기록."""
    def __enter__(self):
        self.peak = list(snap())
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(0.05):
            self.peak = [max((x for x in (a, b) if x is not None), default=None) for a, b in zip(self.peak, snap())]

    def __exit__(self, *_):
        self._stop.set()
        self._t.join()


def recognize(stt, streaming: bool, pcm, chunk: int):
    if not streaming:
        return stt.transcribe(pcm, SR)
    stt.start()
    for i in range(0, len(pcm), chunk):
        stt.feed(pcm[i:i + chunk], SR)
    return stt.finish()


def main(argv=None):
    ap = argparse.ArgumentParser(description="어댑터 적재 전/후 RSS·GPU 메모리와 인식 중 peak를 표로 출력한다.")
    ap.add_argument("--sections", nargs="+", default=["stt.final"], help="예: stt.streaming stt.final llm")
    ap.add_argument("--config", default=str(ROOT / "config.toml"))
    ap.add_argument("--wav", help="있으면 적재 후 한 번 인식하며 peak 측정")
    a = ap.parse_args(argv)

    cfg = tomllib.loads(Path(a.config).read_text(encoding="utf-8"))
    models_dir = cfg.get("models", {}).get("dir", "models")
    pcm = read_wav(a.wav) if a.wav else None
    chunk = int(SR * cfg["audio"]["chunk_ms"] / 1000)
    f = lambda v: "-" if v is None else f"{v:.0f}"
    d = lambda b, c: "-" if b is None or c is None else f"{c - b:+.0f}"
    rows, keep = [], []
    for name in a.sections:
        sec = get_section(cfg, name, a) if name.startswith("stt") else {}
        if "adapter" not in sec:  # llm 등 아직 어댑터 없음
            rows.append([name, "skip"] + ["-"] * 7)
            continue
        streaming = name.endswith("streaming")
        b = snap()
        try:
            with Poll() as p:
                stt = (registry.build_streaming if streaming else registry.build_offline)(sec, models_dir=models_dir)
                stt.load()
                keep.append(stt)
                load_snap = snap()
                if pcm is not None:
                    recognize(stt, streaming, pcm, chunk)
        except NotImplementedError as e:
            rows.append([name, f"skip ({e})"] + ["-"] * 7)
            continue
        rows.append([name, sec["adapter"] + "/" + sec["model"], f(b[0]), f(load_snap[0]), d(b[0], load_snap[0]),
                     f(b[1]), f(load_snap[1]), d(b[1], load_snap[1]), f"{f(p.peak[0])} / {f(p.peak[1])}"])
    head = ["section", "adapter/model", "RSS 전", "RSS 후", "RSS Δ", "GPU 전", "GPU 후", "GPU Δ", "peak RSS / GPU"]
    print("(MiB)\n| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    if gpu_mb() is None:
        print("\nnvidia-smi 없음: GPU 열 생략('-')")


if __name__ == "__main__":
    main()
