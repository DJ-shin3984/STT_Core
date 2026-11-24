"""OutputAdapter: 클립보드 + Ctrl+V (Windows, ctypes).

API (고정):
    out = ClipboardOutput(paste=True, restore_delay_ms=300)
    ok = out.send(text)    # 포커스 창 기록 -> 백업 -> 텍스트 설정 -> 시퀀스 번호 기록 -> (paste) Ctrl+V -> 지연 후 복원
    out.last_note          # 마지막 send의 실패 사유/참고 (없으면 "")
복원은 클립보드 시퀀스 번호가 우리가 설정한 그대로일 때만 한다(사용자가 그 사이 복사했으면 건드리지 않음).
백업이 텍스트가 아니면(이미지·파일) 복원하지 않는다. paste=False는 클립보드에 텍스트만 둔다.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes as w

CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x0002
KEYEVENTF_KEYUP, VK_CONTROL, VK_V = 0x0002, 0x11, 0x56
VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN = 0x12, 0x10, 0x5B, 0x5C

u32 = ctypes.WinDLL("user32", use_last_error=True)
k32 = ctypes.WinDLL("kernel32", use_last_error=True)
P = ctypes.c_void_p


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", w.WORD), ("wScan", w.WORD), ("dwFlags", w.DWORD), ("time", w.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):   # union 크기를 맞추기 위해 필요
    _fields_ = [("dx", w.LONG), ("dy", w.LONG), ("mouseData", w.DWORD), ("dwFlags", w.DWORD),
                ("time", w.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", w.DWORD), ("u", _U)]


for _f, _a, _r in [
    (u32.OpenClipboard, [P], w.BOOL), (u32.CloseClipboard, [], w.BOOL), (u32.EmptyClipboard, [], w.BOOL),
    (u32.GetClipboardData, [w.UINT], P), (u32.SetClipboardData, [w.UINT, P], P),
    (u32.IsClipboardFormatAvailable, [w.UINT], w.BOOL), (u32.CountClipboardFormats, [], ctypes.c_int),
    (u32.GetClipboardSequenceNumber, [], w.DWORD), (u32.GetForegroundWindow, [], P),
    (u32.GetAsyncKeyState, [ctypes.c_int], ctypes.c_short),
    (u32.SendInput, [w.UINT, ctypes.POINTER(INPUT), ctypes.c_int], w.UINT),
    (k32.GlobalAlloc, [w.UINT, ctypes.c_size_t], P), (k32.GlobalLock, [P], P),
    (k32.GlobalUnlock, [P], w.BOOL), (k32.GlobalFree, [P], P)]:
    _f.argtypes, _f.restype = _a, _r


class _Clip:
    """with 블록 동안 클립보드를 연다(5회 재시도, 20ms 간격)."""
    def __enter__(self):
        for _ in range(5):
            if u32.OpenClipboard(None):
                return self
            time.sleep(0.02)
        raise OSError("클립보드를 열 수 없음(다른 프로그램이 사용 중)")

    def __exit__(self, *exc):
        u32.CloseClipboard()


def _get_text():
    """(상태, 텍스트): 'text' | 'empty' | 'other'. 클립보드가 열린 상태에서 호출."""
    if u32.IsClipboardFormatAvailable(CF_UNICODETEXT):
        h = u32.GetClipboardData(CF_UNICODETEXT)
        p = k32.GlobalLock(h) if h else None
        if not p:
            raise OSError("클립보드 텍스트 읽기 실패")
        try:
            return "text", ctypes.wstring_at(p)
        finally:
            k32.GlobalUnlock(h)
    return ("empty" if u32.CountClipboardFormats() == 0 else "other"), None


def _set_text(text):
    """클립보드를 비우고 text를 둔다(None이면 비우기만). 열린 상태에서 호출."""
    if not u32.EmptyClipboard():
        raise OSError("EmptyClipboard 실패")
    if text is None:
        return
    buf = ctypes.create_unicode_buffer(text)
    h = k32.GlobalAlloc(GMEM_MOVEABLE, ctypes.sizeof(buf))
    p = k32.GlobalLock(h) if h else None
    if not p:
        raise OSError("GlobalAlloc/GlobalLock 실패")
    ctypes.memmove(p, buf, ctypes.sizeof(buf))
    k32.GlobalUnlock(h)
    if not u32.SetClipboardData(CF_UNICODETEXT, h):
        k32.GlobalFree(h)
        raise OSError("SetClipboardData 실패")


def _ctrl_v():
    evs = (INPUT * 4)()
    for i, (vk, fl) in enumerate([(VK_CONTROL, 0), (VK_V, 0), (VK_V, KEYEVENTF_KEYUP), (VK_CONTROL, KEYEVENTF_KEYUP)]):
        evs[i].type, evs[i].ki.wVk, evs[i].ki.dwFlags = 1, vk, fl
    if u32.SendInput(4, evs, ctypes.sizeof(INPUT)) != 4:
        raise OSError("SendInput 실패")


def _wait_mods_up(timeout=1.0):
    """단축키의 Alt/Shift/Win이 아직 눌려 있으면 Ctrl+V가 다른 단축키가 되므로 잠시 기다린다."""
    end = time.time() + timeout
    while time.time() < end and any(u32.GetAsyncKeyState(k) & 0x8000 for k in (VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN)):
        time.sleep(0.01)


class ClipboardOutput:
    def __init__(self, paste: bool = True, restore_delay_ms: int = 300):
        self.paste = paste
        self.restore_delay_ms = restore_delay_ms
        self.last_note = ""

    def send(self, text: str) -> bool:
        self.last_note = ""
        try:
            return self._send(text)
        except Exception as e:   # 모든 Windows 호출 실패는 예외 대신 False
            self.last_note = f"출력 실패: {e}"
            return False

    def _send(self, text: str) -> bool:
        fg = u32.GetForegroundWindow()
        with _Clip():
            kind, backup = _get_text()
            _set_text(text)
        seq = u32.GetClipboardSequenceNumber()
        if not self.paste:
            return True
        if kind == "other":
            self.last_note = "기존 클립보드가 텍스트가 아니라(이미지·파일 등) 복원하지 않음"
        ok = True
        try:
            if u32.GetForegroundWindow() != fg:
                ok, self.last_note = False, "포커스 창이 바뀌어 입력을 취소함"
            elif u32.GetClipboardSequenceNumber() != seq:
                ok, self.last_note = False, "클립보드가 바뀌어 입력을 취소함"
            else:
                _wait_mods_up()
                _ctrl_v()
                time.sleep(self.restore_delay_ms / 1000)
        finally:   # Ctrl+V가 실패해도 클립보드는 복원한다
            if kind != "other" and u32.GetClipboardSequenceNumber() == seq:
                with _Clip():
                    _set_text(backup)
        return ok
