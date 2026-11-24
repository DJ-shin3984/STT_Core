import os
import tempfile

from voicecore.vocab import Vocabulary

v = Vocabulary.load("data/vocab/terms_v1.tsv")
assert len(v.terms) == 49, len(v.terms)

hit, miss = 0, []
for c, aliases in v.terms.items():
    gen = v.pronunciations(c)
    assert 1 <= len(gen) <= 4, (c, gen)
    norm = {a.replace(" ", "") for a in aliases}
    if any(g.replace(" ", "") in norm for g in gen):
        hit += 1
    else:
        miss.append((c, gen, aliases))
rate = hit / len(v.terms)
print(f"pronunciation match vs tsv: {hit}/{len(v.terms)} = {rate:.0%}")
for m in miss:
    print("  miss:", m)
assert rate >= 0.8

# aliases_for = tsv ∪ 생성, 중복 없음 / alias_index 길이 내림차순
al = v.aliases_for("PostgreSQL")
assert al[:3] == v.terms["PostgreSQL"] and len(al) == len(set(al))
idx = v.alias_index()
assert all(len(idx[i][0]) >= len(idx[i + 1][0]) for i in range(len(idx) - 1))
assert v.hotwords() == list(v.terms)

# add 후 캐시 무효화, load/save 왕복
v.add("Ktor", ["케이토어"])
assert ("케이토어", "Ktor") in v.alias_index()
path = os.path.join(tempfile.mkdtemp(), "terms.tsv")
v.save(path)
assert Vocabulary.load(path).terms == v.terms
assert Vocabulary.load("nope.tsv").terms == {}
print("test_vocab ok")
