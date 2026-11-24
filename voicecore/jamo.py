"""한글 음절 <-> 자모 분해/조합, 자모 가중 편집거리, 혼동 클래스 골격(skeleton)."""
from __future__ import annotations

from functools import lru_cache

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
        "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")

# 가중치 표 하나. 쌍 키 = 헷갈리기 쉬운 자모(대칭), 문자열 키 = 기본 비용. 나중에 수정 로그로 다시 맞춘다.
WEIGHTS = {
    ("ㅣ", "ㅟ"): 0.3, ("ㅣ", "ㅢ"): 0.3, ("ㅓ", "ㅗ"): 0.4, ("ㅐ", "ㅔ"): 0.2, ("ㅒ", "ㅖ"): 0.2,
    ("ㅚ", "ㅙ"): 0.2, ("ㅚ", "ㅞ"): 0.2, ("ㅙ", "ㅞ"): 0.2, ("ㅡ", "ㅜ"): 0.5,
    ("ㄷ", "ㄸ"): 0.3, ("ㄷ", "ㅌ"): 0.4, ("ㄸ", "ㅌ"): 0.4, ("ㅂ", "ㅍ"): 0.4, ("ㅂ", "ㅃ"): 0.3,
    ("ㄱ", "ㅋ"): 0.4, ("ㄱ", "ㄲ"): 0.3, ("ㄲ", "ㅋ"): 0.4, ("ㅈ", "ㅊ"): 0.4, ("ㅈ", "ㅉ"): 0.3,
    ("ㅅ", "ㅆ"): 0.3, ("ㄹ", "ㄴ"): 0.4,
    "sub": 1.0,        # 같은 자리(초/중/종) 일반 치환
    "sub_kind": 9.0,   # 다른 자리끼리 치환 금지에 가까움
    "indel": 1.0,      # 초성·중성 삽입/삭제
    "indel_jong": 0.5, # 받침 탈락·추가
}
CLASS_MAX = 0.5  # 이 비용 이하 쌍은 같은 혼동 클래스로 묶는다(skeleton용)

_PAIR = {}
for k, v in WEIGHTS.items():
    if isinstance(k, tuple):
        _PAIR[k] = _PAIR[k[::-1]] = v

# 혼동 클래스 대표 문자: union-find
_rep = {}
def _find(c):
    while _rep.get(c, c) != c:
        c = _rep[c]
    return c
for (a, b), v in list(_PAIR.items()):
    if v <= CLASS_MAX:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            _rep[max(ra, rb)] = min(ra, rb)


def is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def decompose(ch: str) -> tuple[str, str, str] | None:
    if not is_hangul(ch):
        return None
    n = ord(ch) - 0xAC00
    return CHO[n // 588], JUNG[n // 28 % 21], JONG[n % 28]


def compose(cho: str, jung: str, jong: str = "") -> str:
    return chr(0xAC00 + (CHO.index(cho) * 21 + JUNG.index(jung)) * 28 + JONG.index(jong))


def normalize_long(s: str) -> str:
    """장음 정규화: 받침 없는 음절 뒤 같은 모음의 'ㅇ+모음' 음절을 지운다(비이트->비트, 지이라->지라)."""
    out = []
    for ch in s:
        d, p = decompose(ch), (decompose(out[-1]) if out else None)
        if d and p and d[0] == "ㅇ" and not d[2] and not p[2] and d[1] == p[1]:
            continue
        out.append(ch)
    return "".join(out)


@lru_cache(maxsize=4096)
def to_jamo(s: str) -> tuple[str, ...]:
    """한글만 자모 토큰열로. 종성은 '_'를 붙여 초성과 구분. 한글 외 문자는 버린다."""
    out = []
    for ch in normalize_long(s):
        d = decompose(ch)
        if d:
            out += [d[0], d[1]] + (["_" + d[2]] if d[2] else [])
    return tuple(out)


def _kind(t: str) -> int:
    return 2 if t[0] == "_" else (1 if t in JUNG else 0)


def _sub(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if _kind(a) != _kind(b):
        return WEIGHTS["sub_kind"]
    return _PAIR.get((a.lstrip("_"), b.lstrip("_")), WEIGHTS["sub"])


def _indel(t: str) -> float:
    return WEIGHTS["indel_jong"] if t[0] == "_" else WEIGHTS["indel"]


@lru_cache(maxsize=65536)
def distance(a: str, b: str) -> float:
    """자모 단위 가중 편집거리(Levenshtein)."""
    x, y = to_jamo(a), to_jamo(b)
    prev = [0.0]
    for t in y:
        prev.append(prev[-1] + _indel(t))
    for s in x:
        cur = [prev[0] + _indel(s)]
        for j, t in enumerate(y, 1):
            cur.append(min(prev[j] + _indel(s), cur[j - 1] + _indel(t), prev[j - 1] + _sub(s, t)))
        prev = cur
    return prev[-1]


def skeleton(s: str) -> tuple[str, ...]:
    """음절마다 (초성 클래스+중성 클래스), 받침 무시. 혼동 쌍 치환·받침 탈락은 같은 골격이 된다."""
    out = []
    for ch in normalize_long(s):
        d = decompose(ch)
        if d:
            out.append(_find(d[0]) + _find(d[1]))
    return tuple(out)


if __name__ == "__main__":
    assert decompose("뷁") == ("ㅂ", "ㅞ", "ㄺ") and compose("ㅂ", "ㅞ", "ㄺ") == "뷁"
    assert normalize_long("비이트") == "비트" and normalize_long("에이피아이") == "에이피아이"
    assert distance("마리아디비", "마리아디비") == 0
    assert distance("마뤼아디비", "마리아디비") == 0.3
    assert distance("레딧", "레디") == 0.5
    assert skeleton("마뤼아뒤비") == skeleton("마리아디비") and skeleton("도커") == skeleton("도꺼")
    print("jamo ok")
