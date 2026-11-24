"""Shell 스모크 테스트: python -m tests.test_shell (오버레이 창이 잠깐 뜬다)"""
import threading
import time

from shell.hotkeys import HotkeyListener, parse
from shell.overlay import Overlay


def test_overlay():
    ov = Overlay()

    def drive():
        ov.set_status("듣는 중")
        ov.show("안녕하세요", " 오늘")
        time.sleep(0.2)
        ov.set_status("처리 중")
        ov.flash("안녕하세요, 오늘", 300)
        ov.set_status("대기")
        time.sleep(0.5)
        ov.close()

    t = threading.Thread(target=drive)
    t.start()
    ov.run()
    t.join()
    print("OK 오버레이 show/set_status/flash/close")


def test_hotkeys():
    assert parse("ctrl+alt+a") == (frozenset({"ctrl", "alt"}), 0x41)
    hl = HotkeyListener({"verbatim": "ctrl+alt+a"}, lambda n: None, lambda n: None)
    hl.start()
    time.sleep(0.5)
    hl.stop()
    # 시뮬레이션: 훅 없이 상태 머신만 점검 (press 1회, 리피트 무시, release 1회)
    ev = []
    hl2 = HotkeyListener({"verbatim": "ctrl+alt+a"}, lambda n: ev.append(("p", n)), lambda n: ev.append(("r", n)))
    for vk, down in [(0xA2, 1), (0xA4, 1), (0x41, 1), (0x41, 1), (0x41, 0), (0x41, 1), (0x41, 0), (0xA4, 0), (0xA2, 0)]:
        hl2._track(vk, bool(down), not down)
    items = []
    while not hl2._cb.empty():
        items.append(hl2._cb.get())
    assert items == [(True, "verbatim"), (False, "verbatim"), (True, "verbatim"), (False, "verbatim")], items
    print("OK 훅 start/stop, press/release 상태 머신")


if __name__ == "__main__":
    test_hotkeys()
    test_overlay()
    print("test_shell 통과")
