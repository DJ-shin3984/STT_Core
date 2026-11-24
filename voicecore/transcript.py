"""TranscriptBuffer: partial → stable(줄지 않음) + unstable(꼬리)."""
from __future__ import annotations

import collections
import os


class TranscriptBuffer:
    def __init__(self, stable_after: int = 3):
        self.n = stable_after
        self.reset()

    def reset(self):
        self.stable, self.unstable = "", ""
        self._hist = collections.deque(maxlen=self.n)

    @property
    def text(self) -> str:
        return self.stable + self.unstable

    def _tail(self, p: str) -> str:
        """partial에서 stable 뒷부분. 앞부분이 고쳐졌으면 stable 어절 수만큼 건너뛴다."""
        if p.startswith(self.stable):
            return p[len(self.stable):]
        return " ".join(p.split()[len(self.stable.split()):])

    def update(self, partial: str) -> tuple[str, str, str]:
        partial = partial.strip()
        self._hist.append(partial)
        new = ""
        if len(self._hist) == self.n:
            p = os.path.commonprefix([self._tail(h) for h in self._hist])
            new = p[:p.rfind(" ") + 1]   # 어절 경계(공백 포함)까지만
            self.stable += new
        self.unstable = self._tail(partial)
        return self.stable, self.unstable, new
