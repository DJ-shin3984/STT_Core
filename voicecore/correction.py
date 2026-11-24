"""Fast correction + safety check.

API (fixed):
    r = correct(text, vocab)            -> CorrectionResult(text, corrections)
    ok, reason = safety_check(original, candidate, vocab)
correct(): exact alias replacement -> jamo-distance candidates -> threshold; keep original when unsure.
safety_check(): for LLM output: numbers/negations preserved, registered terms kept, change ratio bounded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from . import jamo
from .vocab import Vocabulary

# 개발 문맥 신호 단어(문장 안 부분 문자열로 검사)
CONTEXT_WORDS = frozenset("""
서버 설정 포트 배포 컨테이너 커넥션 쿼리 테이블 로그 브랜치 빌드 설치 버전 실행 코드 함수 에러 환경 커밋
클래스 메서드 패키지 의존성 캐시 토큰 인증 엔드포인트 스키마 인덱스 마이그레이션 디버그 디버거 컴포넌트
파이프라인 클러스터 인스턴스 이미지 스크립트 프로세스 레포 리포지토리 레포지토리 파일 데이터베이스 디비
응답 호출 트랜잭션 엔티티 백업 복제 노드 파드 네트워크 프록시 헤더 쿠키 세션 타임아웃 재시작 재기동
워크플로 버킷 리전 시크릿 스레드 메모리 디스크 모듈 라이브러리 프레임워크 플러그인 확장 단축키 인터페이스
타입 변수 필드 배열 객체 컬렉션 스테이징 운영 테스트 리뷰 머지 리베이스 이슈 티켓 스프린트 토픽 파티션 컨슈머
채널 알림 들여쓰기 스코프 문서 접속 문자열 메시지 인덱싱 에이전트
""".split())
PARTICLES = ("에서", "으로", "은", "는", "이", "가", "을", "를", "로", "에", "의", "도", "와", "과", "만")
NEGATION = re.compile(r"안|못|않|말고|아니|없|마라|마세요|지\s*마(?![가-힣])")
NUMBER = re.compile(r"\d+")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
NO_GATE_SYL = 6          # 이 음절 수 이상 별칭은 문맥 없이도 정확 치환(엘라스틱서치 등)
PHON_THRESHOLD = 0.2     # 정규화 자모 거리 상한
PHON_MARGIN = 0.1        # 2위 후보(다른 용어)와의 최소 거리 차
PHON_MIN_SYL, PHON_MAX_SYL = 2, 8
PHON_MAX_WORDS = 3       # 음소 창이 걸칠 수 있는 어절 수(여러 어절이면 별칭의 띄어쓰기 구조와 같아야)
# 일반 개발 대화에도 흔한 단어인 별칭: 다른 등록 용어가 함께 있을 때만 치환
AMBIGUOUS_ALIASES = frozenset({"리스트", "바이트"})
MAX_CHANGE_RATIO = 0.4


@dataclass
class Correction:
    start: int
    end: int
    original: str
    replacement: str
    score: float


@dataclass
class CorrectionResult:
    text: str
    corrections: list[Correction] = field(default_factory=list)


def _syl(s: str) -> int:
    return sum(jamo.is_hangul(c) for c in s)


def _boundary_ok(text: str, start: int, end: int) -> bool:
    """앞은 어절 시작, 뒤는 어절 끝이거나 조사만 붙은 경우."""
    if start > 0 and (text[start - 1].isalnum()):
        return False
    tail = re.match(r"\w*", text[end:]).group()
    return tail == "" or tail in PARTICLES


def _safe(span: str, repl: str) -> bool:
    """숫자는 그대로 남아야 하고 부정어가 걸린 구간은 건드리지 않는다."""
    return not NEGATION.search(span) and all(n in repl for n in NUMBER.findall(span))


def _has_context(text: str, spans) -> bool:
    masked = list(text)
    for s, e in spans:
        masked[s:e] = " " * (e - s)
    masked = "".join(masked)
    return bool(LATIN_WORD.search(masked)) or any(w in masked for w in CONTEXT_WORDS)


def _phon_index(vocab: Vocabulary) -> dict:
    """골격(혼동 클래스) + 음절 1개 삭제 변형 -> [(별칭(공백 제거), 정식)]. SymSpell식 후보 조회용."""
    if "phon" not in vocab._cache:
        idx = {}
        for alias, canon in vocab.alias_index():
            a = alias.replace(" ", "")
            if not re.fullmatch(r"[가-힣]+", a) or _syl(a) < PHON_MIN_SYL or alias in AMBIGUOUS_ALIASES:
                continue
            segs = tuple(len(w) for w in alias.split())
            sk = jamo.skeleton(a)
            for k in {sk} | {sk[:i] + sk[i + 1:] for i in range(len(sk))}:
                idx.setdefault(k, set()).add((a, canon, segs))
        vocab._cache["phon"] = idx
    return vocab._cache["phon"]


def _phonetic(form: str, segs: tuple, idx: dict) -> tuple[str, float] | None:
    """창 하나(segs=어절별 음절 수)에 대해 (정식, 정규화 거리). 임계값·2위 차이를 못 넘으면 None."""
    sk = jamo.skeleton(form)
    cands = set()
    for k in {sk} | {sk[:i] + sk[i + 1:] for i in range(len(sk))}:
        cands |= idx.get(k, set())
    best = {}
    for a, canon, asegs in cands:
        if len(segs) > 1 and asegs != segs:
            continue  # 여러 어절 창은 띄어쓰기 구조가 같은 별칭만(마리아 뒤 비 != 마리아 디비)
        ska = jamo.skeleton(a)
        if len(ska) != len(sk) and (ska[0] != sk[0] or ska[-1] != sk[-1]):
            continue  # 음절 수가 다르면 양끝은 같아야(조사·앞말 먹기 방지)
        d = jamo.distance(form, a) / len(jamo.to_jamo(a))
        best[canon] = min(best.get(canon, 9.0), d)
    ranked = sorted(best.items(), key=lambda kv: kv[1])
    if not ranked or ranked[0][1] > PHON_THRESHOLD:
        return None
    if len(ranked) > 1 and ranked[1][1] - ranked[0][1] < PHON_MARGIN:
        return None
    return ranked[0]


def correct(text: str, vocab: Vocabulary) -> CorrectionResult:
    # 1단계: 정확 별칭 일치(긴 별칭 우선, 겹침 금지, 어절 경계)
    taken = [False] * len(text)
    exact = []
    for alias, canon in vocab.alias_index():
        i = text.find(alias)
        while i != -1:
            j = i + len(alias)
            if not any(taken[i:j]) and _boundary_ok(text, i, j) and _safe(alias, canon):
                exact.append((i, j, canon))
                taken[i:j] = [True] * (j - i)
            i = text.find(alias, i + 1)
    ctx = _has_context(text, [(s, e) for s, e, _ in exact])
    found = []
    for s, e, canon in exact:
        other = any(c != canon for _, _, c in exact)
        if text[s:e] in AMBIGUOUS_ALIASES and not other:
            continue
        if ctx or other or _syl(text[s:e]) >= NO_GATE_SYL:
            found.append(Correction(s, e, text[s:e], canon, 1.0))
    # 2단계: 음소 후보(도메인 게이트: 문맥 신호나 1단계 용어가 있을 때만)
    if ctx or found:
        idx, cand = _phon_index(vocab), []
        words = list(re.finditer(r"\S+", text))
        for a in range(len(words)):
            for b in range(a, min(a + PHON_MAX_WORDS, len(words))):
                chunk = [w.group() for w in words[a:b + 1]]
                if not all(re.fullmatch(r"[가-힣]+", w) for w in chunk):
                    break
                s, e = words[a].start(), words[b].end()
                joined = "".join(chunk)
                forms = [(joined, e)] + [(joined[:-len(p)], e - len(p)) for p in PARTICLES
                                         if joined.endswith(p) and len(joined) > len(p)]
                for form, fe in forms:
                    if not PHON_MIN_SYL <= len(form) <= PHON_MAX_SYL or any(taken[s:fe]) or NEGATION.search(form) \
                            or form in CONTEXT_WORDS:  # 문맥 단어 자체(테스트 등)는 용어 오인식이 아님
                        continue
                    segs = tuple(len(w) for w in chunk[:-1]) + (len(chunk[-1]) - (e - fe),)
                    hit = _phonetic(form, segs, idx)
                    if hit:
                        cand.append((hit[1], -(fe - s), s, fe, hit[0]))
        for d, _, s, fe, canon in sorted(cand):
            if not any(taken[s:fe]):
                taken[s:fe] = [True] * (fe - s)
                found.append(Correction(s, fe, text[s:fe], canon, round(1 - d, 3)))
    found.sort(key=lambda c: c.start)
    out, pos = [], 0
    for c in found:
        out += [text[pos:c.start], c.replacement]
        pos = c.end
    return CorrectionResult("".join(out) + text[pos:], found)


def safety_check(original: str, candidate: str, vocab: Vocabulary) -> tuple[bool, str]:
    if sorted(NUMBER.findall(original)) != sorted(NUMBER.findall(candidate)):
        return False, "numbers changed"
    if len(NEGATION.findall(original)) != len(NEGATION.findall(candidate)):
        return False, "negation changed"
    lost = [t for t in vocab.terms if t in original and t not in candidate]
    if lost:
        return False, f"term dropped: {', '.join(lost)}"
    change = 1 - SequenceMatcher(None, original, candidate, autojunk=False).ratio()
    if change > MAX_CHANGE_RATIO:
        return False, f"change ratio {change:.2f} > {MAX_CHANGE_RATIO}"
    return True, "ok"
