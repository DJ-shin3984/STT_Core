import tempfile
import numpy as np
import soundfile as sf
from evaltools.record import SR, load_manifest, save_take, select_todo, level_msg

S = [{"id": f"s{i}", "category": "trap" if i == 2 else "basic", "speed": "normal", "text": f"문장{i}"} for i in range(1, 4)]
tone = lambda sec: (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(SR * sec)) / SR)).astype(np.float32)

with tempfile.TemporaryDirectory() as d:
    dur, rms, peak = save_take(d, S[0], tone(1.0))
    assert abs(dur - 1.0) < 1e-6 and abs(peak - 0.3) < 0.01 and abs(rms - 0.3 / 2 ** 0.5) < 0.01
    data, sr = sf.read(f"{d}/s1.wav"); assert sr == SR and data.ndim == 1 and len(data) == SR
    assert sf.info(f"{d}/s1.wav").subtype == "PCM_16"
    save_take(d, S[1], tone(0.5))
    m = load_manifest(d); assert [r["id"] for r in m] == ["s1", "s2"]
    save_take(d, S[0], tone(2.0))  # 재녹음 -> 행 교체, 순서 유지
    m = load_manifest(d); assert [r["id"] for r in m] == ["s1", "s2"] and m[0]["duration_sec"] == "2.00"
    assert [s["id"] for s in select_todo(S, m)] == ["s3"]                      # resume
    assert [s["id"] for s in select_todo(S, m, redo=["s2"])] == ["s2"]         # redo
    assert [s["id"] for s in select_todo(S, [], only="trap")] == ["s2"]        # only
    assert list(m[0]) == ["id", "text", "path", "duration_sec", "recorded_at", "category", "speed"]
assert "클리핑" in level_msg(0.1, 1.0) and "너무 작음" in level_msg(0.001, 0.1) and "[" not in level_msg(0.1, 0.5)
print("test_record OK")
