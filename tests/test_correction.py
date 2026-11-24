import re
import statistics
import time

from voicecore import jamo
from voicecore.correction import correct, safety_check
from voicecore.vocab import Vocabulary

v = Vocabulary.load("data/vocab/terms_v1.tsv")
lines = open("data/sentences/sentences_v1.tsv", encoding="utf-8").read().rstrip("\n").split("\n")
rows = [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]
dev = [r for r in rows if r["category"] == "dev"]


def synth(text, terms, alias_of):
    """정식 용어를 한글 별칭으로 바꿔 'STT 출력'을 만든다(긴 용어 먼저, 영문 경계)."""
    for t in sorted(terms, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", alias_of(t), text)
    return text


def restored(out, terms):
    return [t for t in terms if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", out)]


# (a) 긍정: 첫 번째 tsv 별칭으로 합성 -> 복원
ok_terms = n_terms = exact = 0
fails = []
by_split = {"tune": [0, 0], "holdout": [0, 0]}
for r in dev:
    terms = r["terms"].split(";")
    stt = synth(r["text"], terms, lambda t: v.terms[t][0])
    out = correct(stt, v).text
    got = restored(out, terms)
    ok_terms += len(got); n_terms += len(terms)
    by_split[r["split"]][0] += len(got); by_split[r["split"]][1] += len(terms)
    exact += out == r["text"]
    if len(got) < len(terms) or out != r["text"]:
        fails.append((r["id"], stt, out))
rate_a = ok_terms / n_terms
print(f"(a) term restore: {ok_terms}/{n_terms} = {rate_a:.1%}  "
      f"(tune {by_split['tune'][0]}/{by_split['tune'][1]}, holdout {by_split['holdout'][0]}/{by_split['holdout'][1]}); "
      f"sentence exact {exact}/{len(dev)}")
for f in fails:
    print("   ", f)
assert rate_a >= 0.9

# (b) 부정: general + trap 원문은 한 글자도 바뀌면 안 된다
bad = [(r["id"], r["text"], correct(r["text"], v).corrections) for r in rows if r["category"] in ("general", "trap")]
bad = [b for b in bad if b[2]]
print(f"(b) false corrections on general+trap: {len(bad)} sentences")
for b in bad:
    print("   ", b)
assert not bad

# (c) 음소 변형: 별칭 자모 1개를 혼동 쌍으로 바꿈 -> 복원율(보고만)
PAIRS = {a: b for k in jamo.WEIGHTS if isinstance(k, tuple) for a, b in (k, k[::-1])}


def mutate(alias):
    s = list(alias)
    order = list(range(1, len(s))) + [0]          # 첫 음절은 마지막에 시도
    for i in order:
        d = jamo.decompose(s[i])
        if not d:
            continue
        for part in (1, 0, 2):                     # 모음 -> 초성 -> 받침
            if d[part] in PAIRS and (part != 2 or PAIRS[d[part]] in jamo.JONG):
                nd = list(d); nd[part] = PAIRS[d[part]]
                s[i] = jamo.compose(*nd)
                return "".join(s)
    return None


tried = okc = 0
fail_c = []
for r in dev:
    t = r["terms"].split(";")[0]
    alias = v.terms[t][0]
    m = mutate(alias) if sum(map(jamo.is_hangul, alias)) >= 2 else None
    if not m:
        continue
    stt = synth(r["text"], [t], lambda _: m)
    out = correct(stt, v).text
    tried += 1
    if restored(out, [t]):
        okc += 1
    else:
        fail_c.append((stt, out))
    if tried == 20:
        break
print(f"(c) phonetic variant restore: {okc}/{tried} = {okc / tried:.0%}")
for f in fail_c:
    print("   ", f)

# (d) safety_check
o = "Redis 캐시 비우는 건 지금 하지 마 포트 6379"
assert safety_check(o, o, v)[0]
assert safety_check(o, "Redis 캐시 비우는 건 지금 하지 마 포트 6378", v) == (False, "numbers changed")
assert safety_check(o, "Redis 캐시 비우는 건 지금 해 포트 6379", v) == (False, "negation changed")
assert not safety_check("MariaDB 백업은 하지 마", "마리아디비 백업은 하지 마", v)[0]
assert not safety_check(o, "Redis 지금 하지 마 6379 다른 말로 아주 많이 바꿔 버린 문장", v)[0]
assert safety_check("레디스 캐시 비워줘", "Redis 캐시 비워줘.", v)[0]
print("(d) safety_check ok")

# (e) 속도: 25어절 dev 문장, 콜드(캐시 비움) / 웜
long = synth("어제 배포한 Spring Boot 서비스에서 Redis 연결이 간헐적으로 끊기는데 로그 좀 같이 봐줄 수 있어 "
             "그리고 MariaDB 커넥션 풀 설정도 확인하고 Kafka 컨슈머 랙이 얼마나 되는지 알려줘 결과는 슬랙에 남겨줘",
             ["Spring Boot", "Redis", "MariaDB", "Kafka"], lambda t: v.terms[t][0])
long = long.replace("레디스", "레뒤스")
assert len(long.split()) >= 25, len(long.split())
correct(long, v)
cold, warm = [], []
for _ in range(200):
    jamo.distance.cache_clear(); jamo.to_jamo.cache_clear()
    t0 = time.perf_counter(); correct(long, v); cold.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); correct(long, v); warm.append(time.perf_counter() - t0)
p50c, p50w = statistics.median(cold) * 1e3, statistics.median(warm) * 1e3
print(f"(e) correct() {len(long.split())} words, {len(v.alias_index())} aliases: p50 cold {p50c:.2f}ms, warm {p50w:.2f}ms")
print("   ", correct(long, v).text)
assert p50c < 5
print("test_correction ok")
