"""Shell 앱: 단축키 홀드 -> Engine, Engine 이벤트 -> 오버레이.  python -m shell [--config config.toml] [--smoke]"""
from __future__ import annotations

import signal
import threading
import time
import tomllib

from voicecore import events as ev

from .hotkeys import HotkeyListener
from .overlay import Overlay


class FakeEngine:
    """--smoke 용: start 후 0.3초 간격 Partial 3개, stop 후 Finalized + OutputDone."""
    def __init__(self, cfg, on_event):
        self.on_event, self._on = on_event, False

    def load(self):
        time.sleep(0.3)

    def start(self, mode, source=None):
        self._on = True
        threading.Thread(target=self._partials, daemon=True).start()
        return "fake-1"

    def _partials(self):
        for s, u in [("안녕", ""), ("안녕하세요", " 오늘"), ("안녕하세요 오늘", " 날씨가")]:
            if not self._on:
                return
            self.on_event(ev.PartialUpdated("fake-1", s, u))
            time.sleep(0.3)

    def stop(self):
        self._on = False
        self.on_event(ev.UtteranceFinalized("fake-1", "verbatim", "안녕하세요 오늘 날씨가", "안녕하세요, 오늘 날씨가"))
        self.on_event(ev.OutputDone("fake-1", True, "none"))

    def close(self):
        self._on = False


class App:
    def __init__(self, cfg: dict, smoke: bool = False):
        self.cfg, self.smoke = cfg, smoke
        self.ov = Overlay()
        if smoke:
            self.engine = FakeEngine(cfg, self.on_event)
        else:
            from voicecore.engine import Engine
            self.engine = Engine(cfg, self.on_event)
        self.loaded, self.mode = threading.Event(), None
        self.hotkeys = HotkeyListener(cfg["hotkeys"], self.press, self.release)

    def log(self, msg):
        print(f"[shell] {msg}", flush=True)

    def on_event(self, e):
        self.log(f"이벤트 {type(e).__name__}: {getattr(e, 'text', None) or getattr(e, 'message', '') or getattr(e, 'stable', '')}")
        if isinstance(e, ev.PartialUpdated):
            self.ov.show(e.stable, e.unstable)
        elif isinstance(e, ev.UtteranceFinalized):
            self.ov.flash(e.text, 1500)
        elif isinstance(e, ev.OutputDone):
            self.ov.set_status("대기")
        elif isinstance(e, ev.EngineError):
            self.ov.set_status(f"오류: {e.message}")

    def press(self, mode: str):
        if not self.loaded.is_set() or self.mode:
            return
        try:
            self.engine.start(mode)
            self.mode = mode
            self.ov.set_status(f"듣는 중 ({mode})")
        except Exception as e:
            self.ov.set_status(f"오류: {e}")

    def release(self, mode: str):
        if self.mode != mode:
            return
        self.mode = None
        self.ov.set_status("처리 중")
        try:
            self.engine.stop()
        except Exception as e:
            self.ov.set_status(f"오류: {e}")

    def _load(self):
        self.ov.set_status("모델 로딩 중")
        try:
            self.engine.load()
            self.loaded.set()
            self.ov.set_status("대기")
        except Exception as e:
            self.ov.set_status(f"오류: 모델 로딩 실패 {e}")

    def _smoke_driver(self):
        self.loaded.wait(10)
        time.sleep(1)
        self.log("smoke: press")
        self.press("verbatim")
        time.sleep(1.2)
        self.log("smoke: release")
        self.release("verbatim")
        time.sleep(3)
        self.log("smoke: 종료")
        self.ov.close()

    def run(self):
        signal.signal(signal.SIGINT, lambda *_: self.ov.close())
        threading.Thread(target=self._load, daemon=True).start()
        if self.smoke:
            threading.Thread(target=self._smoke_driver, daemon=True).start()
        self.hotkeys.start()
        try:
            self.ov.run()
        finally:
            self.hotkeys.stop()
            self.engine.close()
            self.log("정리 완료")


def main(config="config.toml", smoke=False):
    with open(config, "rb") as f:
        App(tomllib.load(f), smoke).run()
