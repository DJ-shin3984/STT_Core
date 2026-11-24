"""받아쓰기 오버레이 (tkinter). 어느 스레드에서 불러도 안전: 명령을 queue에 넣고 Tk 스레드가 30ms마다 비운다.

    ov = Overlay(); ov.show(stable, unstable); ov.set_status("듣는 중"); ov.flash("최종 텍스트", 1500)
    ov.hide(); ov.run()   # run()은 Overlay를 만든 스레드에서 (mainloop)
상태가 "대기"이고 flash 중이 아니면 창을 숨긴다. 포커스를 뺏으면 붙여넣기가 깨지므로 NOACTIVATE + 클릭 통과로 만든다.
"""
from __future__ import annotations

import ctypes
import queue
import time
import tkinter as tk

FONT = "Malgun Gothic"


class Overlay:
    def __init__(self):
        self.root = r = tk.Tk()
        r.withdraw()
        r.overrideredirect(True)
        r.attributes("-topmost", True, "-alpha", 0.9)
        r.configure(bg="#f5f5f5")
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        width, height = int(sw * 0.6), 96
        r.geometry(f"{width}x{height}+{(sw - width) // 2}+{sh - height - 80}")
        self.status = tk.Label(r, text="대기", bg="#f5f5f5", fg="#2a6", font=(FONT, 10), anchor="w")
        self.status.pack(fill="x", padx=14, pady=(6, 0))
        self.text = tk.Text(r, bg="#f5f5f5", relief="flat", height=2, wrap="word", font=(FONT, 16), cursor="arrow")
        self.text.tag_config("stable", foreground="#111", font=(FONT, 16, "bold"))
        self.text.tag_config("unstable", foreground="#999")
        self.text.pack(fill="both", expand=True, padx=14, pady=(0, 6))
        self._q: queue.Queue = queue.Queue()
        self._flash_until = 0.0
        self._closed = False
        r.update_idletasks()
        self._no_activate()
        r.after(30, self._poll)

    # ---- 어느 스레드에서나 ----
    def show(self, stable: str, unstable: str = ""):
        self._q.put(("show", stable, unstable))

    def set_status(self, text: str):
        self._q.put(("status", text))

    def flash(self, text: str, ms: int = 1500):
        self._q.put(("flash", text, ms))

    def hide(self):
        self._q.put(("hide",))

    def close(self):
        self._q.put(("close",))

    def run(self):
        self.root.mainloop()

    # ---- Tk 스레드 ----
    def _no_activate(self):
        u = ctypes.windll.user32
        u.GetParent.restype = ctypes.c_void_p
        u.GetParent.argtypes = [ctypes.c_void_p]
        u.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        hwnd = u.GetParent(self.root.winfo_id()) or self.root.winfo_id()
        # WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT(클릭 통과)
        u.SetWindowLongW(hwnd, -20, u.GetWindowLongW(hwnd, -20) | 0x80 | 0x08000000 | 0x20)

    def _set_text(self, stable, unstable=""):
        self.text.delete("1.0", "end")
        self.text.insert("end", stable, "stable")
        self.text.insert("end", unstable, "unstable")

    def _visible(self, on: bool):
        self.root.deiconify() if on else self.root.withdraw()
        if on:
            self.root.attributes("-topmost", True)

    def _idle_hide(self):
        if self.status.cget("text") == "대기" and time.time() >= self._flash_until:
            self._set_text("")
            self._visible(False)

    def _poll(self):
        while not self._closed:
            try:
                cmd = self._q.get_nowait()
            except queue.Empty:
                break
            op = cmd[0]
            if op == "show":
                self._set_text(cmd[1], cmd[2])
                self._visible(True)
            elif op == "status":
                self.status.config(text=cmd[1])
                self._visible(True) if cmd[1] != "대기" else self._idle_hide()
            elif op == "flash":
                self._set_text(cmd[1])
                self._flash_until = time.time() + cmd[2] / 1000
                self._visible(True)
                self.root.after(cmd[2] + 50, self._idle_hide)
            elif op == "hide":
                self._visible(False)
            elif op == "close":
                self._closed = True
                self.root.destroy()
        if not self._closed:
            self.root.after(30, self._poll)
