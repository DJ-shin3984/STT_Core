"""문장을 보여 주고 녹음해 data/recordings/에 저장한다.
사용: python -m evaltools.record --sentences data/sentences/sentences_v1.tsv"""
import argparse, csv, sys, tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
COLS = ["id", "text", "path", "duration_sec", "recorded_at", "category", "speed"]


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_sentences(path):
    if not Path(path).is_file():
        sys.exit(f"오류: 문장 파일이 없습니다: {path}")
    return read_tsv(path)


def load_manifest(out):
    p = Path(out) / "manifest.tsv"
    return read_tsv(p) if p.is_file() else []


def select_todo(sentences, manifest, redo=(), only=None):
    """녹음할 문장 목록. redo가 있으면 그 id만, 아니면 manifest에 없는 것만."""
    done = {r["id"] for r in manifest}
    todo = [s for s in sentences if (s["id"] in redo if redo else s["id"] not in done)]
    return [s for s in todo if not only or s["category"] == only]


def save_take(out, s, audio):
    """wav 저장 + manifest 행 교체/추가. (duration, rms, peak) 반환."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    wav = out / f"{s['id']}.wav"
    sf.write(wav, audio, SR, subtype="PCM_16")
    dur = len(audio) / SR
    row = {"id": s["id"], "text": s["text"], "path": wav.as_posix(), "duration_sec": f"{dur:.2f}",
           "recorded_at": datetime.now().isoformat(timespec="seconds"),
           "category": s["category"], "speed": s["speed"]}
    rows = [r for r in load_manifest(out) if r["id"] != s["id"]]
    pos = next((i for i, r in enumerate(load_manifest(out)) if r["id"] == s["id"]), len(rows))
    rows.insert(pos, row)
    with open(out / "manifest.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t"); w.writeheader(); w.writerows(rows)
    return dur, float(np.sqrt(np.mean(audio ** 2))), float(np.max(np.abs(audio)))


def level_msg(rms, peak):
    msg = f"RMS {rms:.4f}, peak {peak:.3f}"
    if peak >= 0.99: msg += "  [경고] 클리핑"
    if rms < 0.005: msg += "  [경고] 너무 작음"
    return msg


def pick_device(arg):
    if arg is None:
        try:
            with open("config.toml", "rb") as f:
                arg = tomllib.load(f).get("audio", {}).get("input_device", "")
        except FileNotFoundError:
            arg = ""
    if not arg: return None
    return int(arg) if str(arg).isdigit() else str(arg)


def record_until_enter(device):
    import sounddevice as sd
    blocks = []
    try:
        st = sd.InputStream(samplerate=SR, channels=1, dtype="float32", device=device,
                            callback=lambda d, n, t, s: blocks.append(d.copy()))
        st.start()
    except sd.PortAudioError as e:
        info = sd.query_devices(device if device is not None else sd.default.device[0])
        sys.exit(f"오류: 장치가 16kHz 녹음을 거부했습니다 ({e})\n"
                 f"  장치: {info['name']}, 기본 샘플레이트: {info['default_samplerate']} Hz (리샘플링은 지원하지 않음)")
    input("  ● 녹음 중... Enter로 정지")
    st.stop(); st.close()
    return np.concatenate(blocks)[:, 0] if blocks else np.zeros(0, np.float32)


def selftest(device):
    import sounddevice as sd
    a = sd.rec(SR, samplerate=SR, channels=1, dtype="float32", device=device); sd.wait()
    a = a[:, 0]
    print("1초 자가 녹음:", level_msg(float(np.sqrt(np.mean(a ** 2))), float(np.max(np.abs(a)))))


def main():
    ap = argparse.ArgumentParser(description="문장 녹음 도구")
    ap.add_argument("--sentences"); ap.add_argument("--out", default="data/recordings")
    ap.add_argument("--device"); ap.add_argument("--only", help="category 필터 (예: trap)")
    ap.add_argument("--redo", help="다시 녹음할 id (쉼표 구분)")
    ap.add_argument("--list-devices", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    import sounddevice as sd
    if a.list_devices: print(sd.query_devices()); return
    dev = pick_device(a.device)
    if a.selftest: selftest(dev); return
    if not a.sentences: ap.error("--sentences 가 필요합니다")

    sents = load_sentences(a.sentences)
    redo = [x for x in (a.redo or "").split(",") if x]
    todo = select_todo(sents, load_manifest(a.out), redo, a.only)
    total, n_rec, n_skip, secs = len(sents), 0, 0, 0.0
    ids = [s["id"] for s in sents]
    print(f"녹음할 문장 {len(todo)}개 (전체 {total}개)")
    for s in todo:
        audio = None
        while True:
            print(f"\n[{ids.index(s['id']) + 1}/{total}] {s['id']}  ({s['category']}, {s['speed']})")
            if s["speed"] == "fast": print("  >>> 빠르게 읽기 <<<")
            print(f"\n    {s['text']}\n")
            input("  Enter = 녹음 시작")
            audio = record_until_enter(dev)
            while True:
                c = input("  Enter=저장 후 다음 / r=다시 / p=재생 / s=건너뛰기 / q=종료 > ").strip().lower()
                if c == "p": sd.play(audio, SR); sd.wait()
                else: break
            if c != "r": break
        if c == "q": break
        if c == "s": n_skip += 1; continue
        dur, rms, peak = save_take(a.out, s, audio)
        n_rec += 1; secs += dur
        print(f"  저장됨 {dur:.1f}초, {level_msg(rms, peak)}")
    print(f"\n요약: 녹음 {n_rec}개, 건너뜀 {n_skip}개, 총 길이 {secs:.1f}초")


if __name__ == "__main__":
    main()
