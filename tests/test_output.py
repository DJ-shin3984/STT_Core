"""ClipboardOutput 테스트: python -m tests.test_output

자동 테스트는 paste=False만 검사한다(paste=True는 포커스 창에 실제로 붙여넣기가 일어나므로 실행하지 않음).
수동 절차(paste=True): 메모장을 열어 커서를 두고, 다른 창에서 아래를 실행한 뒤 3초 안에 메모장을 클릭한다.
    .venv\\Scripts\\python.exe -X utf8 -c "import time; from voicecore.output import ClipboardOutput as C; \
time.sleep(3); o=C(paste=True); print(o.send('수동 테스트 text'), o.last_note)"
기대: 메모장에 문장이 붙고 True 출력, 실행 전 클립보드 텍스트가 300ms 뒤 복원됨. 대기 중 다른 창을 클릭하면 False.
"""
from voicecore.output import _Clip, _get_text, _set_text, ClipboardOutput


def clip_text():
    with _Clip():
        return _get_text()


def main():
    kind, orig = clip_text()
    try:
        out = ClipboardOutput(paste=False)
        assert out.send("테스트 한글 text") is True, out.last_note
        assert clip_text() == ("text", "테스트 한글 text")
        assert out.last_note == ""
        print("OK 클립보드 설정 확인")
    finally:
        if kind == "text":
            with _Clip():
                _set_text(orig)
        print(f"원래 클립보드({kind}) 복원")
    if kind == "text":
        assert clip_text() == ("text", orig)
    print("test_output 통과")


if __name__ == "__main__":
    main()
