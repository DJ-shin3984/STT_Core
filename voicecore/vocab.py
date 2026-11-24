"""VocabularyManager + PronunciationGenerator.

API (fixed):
    v = Vocabulary.load("data/vocab/terms.tsv")   # missing file -> empty vocab
    v.terms            -> dict[str, list[str]]     canonical -> aliases (Korean spellings etc.)
    v.hotwords()       -> list[str]                canonical terms for STT biasing
    v.add("MariaDB", ["마리아디비"]); v.save(path)
    v.pronunciations("MariaDB") -> list[str]       generated Korean spellings (PronunciationGenerator)
    v.aliases_for("MariaDB")    -> list[str]       tsv aliases ∪ generated (deduped)
    v.alias_index()             -> list[(alias, canonical)], alias length desc, cached
tsv format: header `canonical<TAB>aliases` (aliases separated by semicolons).
"""
from __future__ import annotations

import csv
import itertools
import os
import re
from functools import lru_cache

# 기술 토큰 -> 한글 발음('|'로 변형). 소문자 키. 여러 조각 합친 키(github)가 우선.
TOKENS = {k: v.split("|") for k, v in (l.split("=") for l in """
maria=마리아 mongo=몽고 postgresql=포스트그레스큐엘|포스트그레에스큐엘 postgre=포스트그레 postgres=포스트그레스 mysql=마이에스큐엘 oracle=오라클
redis=레디스 kafka=카프카 rabbit=래빗 mq=엠큐 elastic=엘라스틱 search=서치 elasticsearch=엘라스틱서치|일래스틱서치
kibana=키바나 logstash=로그스태시 grafana=그라파나 prometheus=프로메테우스 sentry=센트리 datadog=데이터독
docker=도커 compose=컴포즈 kubernetes=쿠버네티스 helm=헬름 nginx=엔진엑스 apache=아파치 tomcat=톰캣
jenkins=젠킨스 github=깃허브|깃헙 gitlab=깃랩 git=깃|기트 hub=허브 actions=액션즈|액션 action=액션 bitbucket=비트버킷
spring=스프링 boot=부트 security=시큐리티 cloud=클라우드 batch=배치 data=데이터 hibernate=하이버네이트
jpa=제이피에이 mybatis=마이바티스 java=자바 kotlin=코틀린 python=파이썬 django=장고 flask=플라스크
fastapi=패스트에이피아이 node=노드 nodejs=노드제이에스 js=제이에스 express=익스프레스 nest=네스트 next=넥스트
react=리액트 redux=리덕스 vue=뷰 vuex=뷰엑스 angular=앵귤러 svelte=스벨트 vite=비트|바이트 webpack=웹팩
babel=바벨 eslint=이에스린트 prettier=프리티어 type=타입 script=스크립트 typescript=타입스크립트
javascript=자바스크립트 html=에이치티엠엘 css=씨에스에스|시에스에스 json=제이슨 yaml=야믈|야멜 xml=엑스엠엘
toml=톰엘 rest=레스트 graphql=그래프큐엘 rpc=알피씨 http=에이치티티피 https=에이치티티피에스 api=에이피아이
oauth=오어스|오오스 auth=어스 jwt=제이더블유티 token=토큰 sql=에스큐엘|시퀄 db=디비 nosql=노에스큐엘
aws=에이더블유에스 ec2=이씨투 s3=에스쓰리 lambda=람다 gcp=지씨피 azure=애저 terraform=테라폼 ansible=앤서블
linux=리눅스 ubuntu=우분투 centos=센트오에스 windows=윈도우 mac=맥 bash=배시 shell=셸|쉘
vs=브이에스|비에스 code=코드 visual=비주얼 studio=스튜디오 intellij=인텔리제이 intelli=인텔리 eclipse=이클립스
postman=포스트맨 post=포스트 man=맨 swagger=스웨거 jira=지라 confluence=컨플루언스 slack=슬랙 notion=노션
figma=피그마 gradle=그래들|그레이들 maven=메이븐 npm=엔피엠 yarn=얀 pnpm=피엔피엠 pip=핍 conda=콘다
junit=제이유닛 jest=제스트 mockito=모키토 selenium=셀레니움 cypress=사이프러스 pytest=파이테스트
ci=씨아이|시아이 cd=씨디|시디 pr=피알 mr=엠알 ide=아이디이 sdk=에스디케이 cli=씨엘아이 ui=유아이 ux=유엑스
url=유알엘 uri=유알아이 dns=디엔에스 ssl=에스에스엘 tls=티엘에스 ssh=에스에스에이치 tcp=티씨피 udp=유디피
ip=아이피 cpu=씨피유 gpu=지피유 ram=램 ssd=에스에스디 os=오에스 vm=브이엠 k8s=케이에이츠
cache=캐시 server=서버 client=클라이언트 config=컨피그 log=로그 test=테스트 build=빌드 deploy=디플로이
grpc=지알피씨 go=고 rust=러스트 ruby=루비 rails=레일즈 swift=스위프트 php=피에이치피 laravel=라라벨
unity=유니티 flutter=플러터 dart=다트 android=안드로이드 ios=아이오에스 firebase=파이어베이스
supabase=수파베이스 vercel=버셀 netlify=네틀리파이 heroku=헤로쿠 openai=오픈에이아이 llama=라마
""".split())}
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "에이 비 씨 디 이 에프 지 에이치 아이 제이 케이 엘 엠 엔 오 피 큐 알 에스 티 유 브이 더블유 엑스 와이 지".split()))
DIGITS = dict(zip("0123456789", "제로 원 투 쓰리 포 파이브 식스 세븐 에잇 나인".split()))
MAX_PRON = 4

_PIECE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# 모르는 단어용 단순 자소 규칙(근사). 완전한 G2P 아님.
_ONSET = {"b": "ㅂ", "c": "ㅋ", "d": "ㄷ", "f": "ㅍ", "g": "ㄱ", "h": "ㅎ", "j": "ㅈ", "k": "ㅋ", "l": "ㄹ", "m": "ㅁ",
          "n": "ㄴ", "p": "ㅍ", "q": "ㅋ", "r": "ㄹ", "s": "ㅅ", "t": "ㅌ", "v": "ㅂ", "w": "ㅇ", "x": "ㅅ", "z": "ㅈ",
          "ch": "ㅊ", "sh": "ㅅ", "th": "ㅅ", "ph": "ㅍ", "ck": "ㅋ", "qu": "ㅋ"}
_CODA = {"k": "ㄱ", "ck": "ㄱ", "c": "ㄱ", "g": "ㄱ", "n": "ㄴ", "l": "ㄹ", "m": "ㅁ", "p": "ㅂ", "b": "ㅂ", "t": "ㅅ", "ng": "ㅇ"}
_VOWEL = {"a": "ㅏ", "e": "ㅔ", "i": "ㅣ", "o": "ㅗ", "u": "ㅓ", "y": "ㅣ", "ee": "ㅣ", "ea": "ㅣ", "oo": "ㅜ",
          "ai": "ㅔ", "ay": "ㅔ", "ou": "ㅏ", "ow": "ㅏ", "er": "ㅓ", "ar": "ㅏ", "or": "ㅗ", "ie": "ㅣ"}


def _rule_g2p(word: str) -> str:
    from .jamo import compose
    w = re.sub(r"(?<=[^aeiou])e$", "", word.lower())          # 끝 묵음 e
    parts = re.findall(r"ee|ea|oo|ai|ay|ou|ow|ie|[aeiouy]r(?![aeiouy])|[aeiouy]|ng|ch|sh|th|ph|ck|qu|[a-z]", w)
    out, onset = [], "ㅇ"
    for i, p in enumerate(parts):
        if p in _VOWEL or p[0] in "aeiouy":
            out.append([onset, _VOWEL.get(p, _VOWEL.get(p[0], "ㅓ")), ""]); onset = "ㅇ"
            continue
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if nxt and (nxt in _VOWEL or nxt[0] in "aeiouy"):
            if onset != "ㅇ":
                out.append([onset, "ㅡ", ""])
            onset = _ONSET.get(p, "ㅇ")
        elif out and not out[-1][2] and onset == "ㅇ" and p in _CODA:
            out[-1][2] = _CODA[p]
        elif p != "r":                                         # 모음 뒤 r은 묵음
            if onset != "ㅇ":
                out.append([onset, "ㅡ", ""])
            onset = _ONSET.get(p, "ㅇ")
    if onset != "ㅇ":
        out.append([onset, "ㅡ", ""])
    return "".join(compose(*s) for s in out)


def _read(piece: str) -> tuple[list[str], bool]:
    """조각 하나의 읽기 후보와 '약어(대문자)' 여부."""
    low = piece.lower()
    if low in TOKENS:
        return TOKENS[low], piece.isupper() and len(piece) > 1
    if piece.isdigit():
        return ["".join(DIGITS[d] for d in piece)], False
    if piece.isupper() or len(piece) == 1:
        return ["".join(LETTERS[c] for c in piece.upper())], len(piece) > 1
    return [_rule_g2p(piece)], False


@lru_cache(maxsize=1024)
def generate(canonical: str) -> tuple[str, ...]:
    """영문 용어 -> 한글 발음 후보 1~MAX_PRON개."""
    words = []                                                  # [[(읽기들, 약어?, 원문조각)]]
    for word in re.split(r"[\s/]+", canonical.strip()):
        if not word:
            continue
        pieces = _PIECE.findall(word.replace(".", " "))
        merged, i = [], 0
        while i < len(pieces):                                  # 인접 조각을 합친 키가 사전에 있으면 우선(Git+Hub)
            for j in range(len(pieces), i, -1):
                key = "".join(pieces[i:j])
                if j - i > 1 and key.lower() in TOKENS:
                    merged.append(key); i = j; break
            else:
                merged.append(pieces[i]); i += 1
        words.append([(*_read(p), p) for p in merged])
    flat = [x for w in words for x in w]
    if not flat:
        return ()
    first = [r[0] for r, _, _ in flat]
    out = ["".join(first),                                       # 마리아디비
           " ".join("".join(r[0] for r, _, _ in w) for w in words),  # 단어 경계 공백
           " ".join(first)]                                      # 조각 공백: 마리아 디비
    for i, (r, _, _) in enumerate(flat):                         # 두 번째 읽기(시퀄, 깃헙)
        for alt in r[1:2]:
            out.append("".join(first[:i] + [alt] + first[i + 1:]))
    if any(a for _, a, _ in flat) and not all(a for _, a, _ in flat):  # 마리아 DB
        out.append(" ".join(p if a else r[0] for r, a, p in flat))
    seen = []
    for s in out:
        if s not in seen and s != canonical:
            seen.append(s)
    return tuple(seen[:MAX_PRON])


class Vocabulary:
    def __init__(self, terms: dict[str, list[str]] | None = None):
        self.terms = terms or {}
        self._cache = {}

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        if not path or not os.path.exists(path):
            return cls()
        terms = {}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                terms[row["canonical"]] = [a for a in (row.get("aliases") or "").split(";") if a]
        return cls(terms)

    def hotwords(self) -> list[str]:
        return list(self.terms)

    def add(self, canonical: str, aliases=()):
        self.terms.setdefault(canonical, [])
        for a in aliases:
            if a not in self.terms[canonical]:
                self.terms[canonical].append(a)
        self._cache.clear()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("canonical\taliases\n")
            for c, al in self.terms.items():
                f.write(f"{c}\t{';'.join(al)}\n")

    def pronunciations(self, canonical: str) -> list[str]:
        return list(generate(canonical))

    def aliases_for(self, canonical: str) -> list[str]:
        return list(dict.fromkeys(a for a in self.terms.get(canonical, []) + self.pronunciations(canonical)
                                  if a and a != canonical))

    def alias_index(self) -> list[tuple[str, str]]:
        if "index" not in self._cache:
            pairs = {(a, c) for c in self.terms for a in self.aliases_for(c)}
            self._cache["index"] = sorted(pairs, key=lambda p: (-len(p[0]), p[0]))
        return self._cache["index"]
