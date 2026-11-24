"""TranscriptBuffer 테스트: python -m tests.test_transcript"""
from __future__ import annotations

from voicecore.transcript import TranscriptBuffer


def run(seq, n=3):
    tb, acc, prev = TranscriptBuffer(n), "", ""
    for p in seq:
        stable, unstable, new = tb.update(p)
        assert stable.startswith(prev), f"stable 줄어듦: {prev!r} -> {stable!r}"
        assert stable == "" or stable.endswith(" "), f"어절 경계 아님: {stable!r}"
        acc += new
        assert acc == stable, (acc, stable)
        assert tb.text == stable + unstable
        prev = stable
    return tb


def test_growing_tail():
    target = "부모가 저지르는 큰 실수 중 하나는"
    tb = run([target[:i] for i in range(1, len(target) + 1)])
    assert tb.stable == "부모가 저지르는 큰 실수 중 ", tb.stable
    assert tb.unstable == "하나는"


def test_changing_tail():
    tb = run(["그는 괜", "그는 괜찬", "그는 괜찮은", "그는 괜찮은 척", "그는 괜찮은 첰", "그는 괜찮은 척하"])
    assert tb.stable == "그는 괜찮은 ", tb.stable     # 어절 중간에서 자르지 않음
    assert tb.unstable == "척하"


def test_front_revised():
    tb = run(["지하철 에서", "지하철 에서 다리", "지하철 에서 다리를"])
    assert tb.stable == "지하철 ", tb.stable          # 마지막 어절이 끝났는지 모르므로 "에서"는 아직
    # stable 이 된 뒤 앞부분이 고쳐져도("에서"→"애서") stable은 그대로, 꼬리는 어절 수만큼 건너뛴 뒤
    tb = run(["지하철 에서 다", "지하철 에서 다리", "지하철 에서 다리를",
              "지하철 애서 다리를 벌리고", "지하철 애서 다리를 벌리고 앉지", "지하철 애서 다리를 벌리고 앉지 마라"])
    assert tb.stable == "지하철 에서 다리를 ", tb.stable
    assert tb.unstable == "벌리고 앉지 마라", tb.unstable


def test_empty():
    tb = run(["", "", "", "안녕", "", "", ""])
    assert tb.stable == "" and tb.unstable == ""
    tb.reset()
    assert tb.text == ""
    run(["가 나", "가 나", "가 나", "", "", ""])   # 빈 partial 이 와도 stable 유지


def main():
    for f in (test_growing_tail, test_changing_tail, test_front_revised, test_empty):
        f()
        print(f"{f.__name__}: OK")


if __name__ == "__main__":
    main()
