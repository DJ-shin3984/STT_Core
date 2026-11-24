"""아주 작은 구간 시간 기록기. 이름 붙인 시각을 perf_counter로 남긴다."""
import time


class Tracer:
    def __init__(self):
        self.t: dict[str, float] = {}

    def mark(self, name: str) -> float:
        self.t[name] = time.perf_counter()
        return self.t[name]

    def elapsed_ms(self, a: str, b: str) -> float:
        return (self.t[b] - self.t[a]) * 1000

    def to_dict(self) -> dict[str, float]:
        """첫 mark 기준 상대 시각(ms)."""
        t0 = min(self.t.values(), default=0.0)
        return {k: round((v - t0) * 1000, 3) for k, v in self.t.items()}
