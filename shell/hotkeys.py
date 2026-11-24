"""전역 단축키 홀드 감지 (WH_KEYBOARD_LL). 키는 삼키지 않는다(패스스루).

    hl = HotkeyListener({"verbatim": "ctrl+alt+a"}, on_press=f(name), on_release=f(name))
    hl.start(); ...; hl.stop()
훅 콜백은 빨라야 하므로(느리면 Windows가 훅을 제거) press/release 콜백은 별도 스레드에서 순서대로 부른다.
자가 테스트: python -m shell.hotkeys --test --seconds 5
"""
from __future__ import annotations

import ctypes
import queue
import threading
from ctypes import wintypes as w

u32 = ctypes.WinDLL("user32", use_last_error=True)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
WH_KEYBOARD_LL, WM_QUIT, LLKHF_INJECTED = 13, 0x0012, 0x10
DOWN, UP = (0x100, 0x104), (0x101, 0x105)
MODS = {"ctrl": {0x11, 0xA2, 0xA3}, "alt": {0x12, 0xA4, 0xA5}, "shift": {0x10, 0xA0, 0xA1}, "win": {0x5B, 0x5C}}
NAMED = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "esc": 0x1B, **{f"f{i}": 0x6F + i for i in range(1, 13)}}
LRESULT = ctypes.c_ssize_t
PROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, w.WPARAM, w.LPARAM)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", w.DWORD), ("scanCode", w.DWORD), ("flags", w.DWORD), ("time", w.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


u32.SetWindowsHookExW.argtypes = [ctypes.c_int, PROC, w.HINSTANCE, w.DWORD]
u32.SetWindowsHookExW.restype = ctypes.c_void_p
u32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, w.WPARAM, w.LPARAM]
u32.CallNextHookEx.restype = LRESULT
u32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
u32.GetMessageW.argtypes = [ctypes.POINTER(w.MSG), w.HWND, w.UINT, w.UINT]
u32.PeekMessageW.argtypes = [ctypes.POINTER(w.MSG), w.HWND, w.UINT, w.UINT, w.UINT]
u32.PostThreadMessageW.argtypes = [w.DWORD, w.UINT, w.WPARAM, w.LPARAM]
k32.GetModuleHandleW.argtypes = [w.LPCWSTR]
k32.GetModuleHandleW.restype = w.HINSTANCE


def _mod_of(vk):
    return next((m for m, s in MODS.items() if vk in s), None)


def parse(combo: str):
    """'ctrl+alt+a' -> (frozenset({'ctrl','alt'}), 0x41)"""
    parts = [p.strip().lower() for p in combo.split("+")]
    mods, key = frozenset(p for p in parts if p in MODS), [p for p in parts if p not in MODS]
    if len(key) != 1:
        raise ValueError(f"단축키 형식 오류: {combo!r} (수식키 + 키 하나)")
    k = key[0]
    vk = NAMED.get(k) or (ord(k.upper()) if len(k) == 1 and k.isalnum() else None)
    if vk is None:
        raise ValueError(f"알 수 없는 키: {k!r}")
    return mods, vk


class HotkeyListener:
    def __init__(self, bindings: dict, on_press, on_release):
        self.binds = {name: parse(c) for name, c in bindings.items()}
        self.on_press, self.on_release = on_press, on_release
        self._held, self._mods, self._active = set(), set(), None   # 눌린 일반 키 / 수식키 / 눌려 있는 조합 이름
        self._cb: queue.Queue = queue.Queue()
        self._ready, self._tid, self._err = threading.Event(), 0, None
        self._threads = []

    def start(self):
        self._threads = [threading.Thread(target=self._hook_loop, name="hotkey-hook", daemon=True),
                         threading.Thread(target=self._dispatch, name="hotkey-callbacks", daemon=True)]
        for t in self._threads:
            t.start()
        if not self._ready.wait(3) or self._err:
            raise RuntimeError(self._err or "훅 설치 시간 초과")

    def stop(self):
        if self._tid:
            u32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
        self._cb.put(None)
        for t in self._threads:
            t.join(2)
        self._tid = 0

    def _dispatch(self):
        while (item := self._cb.get()) is not None:
            try:
                (self.on_press if item[0] else self.on_release)(item[1])
            except Exception as e:
                print(f"[hotkeys] 콜백 오류: {e}")

    def _hook_loop(self):
        msg = w.MSG()
        u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)   # 이 스레드의 메시지 큐 생성
        self._tid = k32.GetCurrentThreadId()
        proc = PROC(self._on_key)                             # 참조를 유지해야 GC되지 않는다
        hook = u32.SetWindowsHookExW(WH_KEYBOARD_LL, proc, k32.GetModuleHandleW(None), 0)
        self._err = None if hook else f"SetWindowsHookExW 실패 (오류 {ctypes.get_last_error()})"
        self._ready.set()
        if hook:
            while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                pass
            u32.UnhookWindowsHookEx(hook)

    def _on_key(self, code, wparam, lparam):
        try:
            if code >= 0:
                k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if not k.flags & LLKHF_INJECTED:              # 우리가 보낸 Ctrl+V 등은 무시
                    self._track(k.vkCode, wparam in DOWN, wparam in UP)
        except Exception:
            pass
        return u32.CallNextHookEx(None, code, wparam, lparam)   # 항상 패스스루

    def _track(self, vk, down, up):
        mod = _mod_of(vk)
        group = self._mods if mod else self._held
        item = mod or vk
        if down:
            group.add(item)
            if self._active is None:                          # 키 리피트는 이미 active라 무시된다
                for name, (mods, key) in self.binds.items():
                    if key in self._held and mods <= self._mods:
                        self._active = name
                        self._cb.put((True, name))
                        break
        elif up:
            group.discard(item)
            if self._active is not None:
                mods, key = self.binds[self._active]
                if not (key in self._held and mods <= self._mods):
                    self._cb.put((False, self._active))
                    self._active = None


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--seconds", type=float, default=5)
    a = ap.parse_args()
    hl = HotkeyListener({"verbatim": "ctrl+alt+a", "polished": "ctrl+alt+b"},
                        lambda n: print(f"press   {n}", flush=True), lambda n: print(f"release {n}", flush=True))
    hl.start()
    print(f"훅 설치됨. {a.seconds}초 동안 ctrl+alt+a / ctrl+alt+b 를 눌러 보세요.", flush=True)
    time.sleep(a.seconds)
    hl.stop()
    print("훅 해제됨.")
