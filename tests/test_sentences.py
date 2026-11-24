import csv, collections, re
S = "data/sentences/sentences_v1.tsv"
V = "data/vocab/terms_v1.tsv"
raw = open(S, encoding="utf-8").read()
assert not raw.startswith("﻿") and '"' not in raw and "\r" not in raw
lines = raw.rstrip("\n").split("\n")
assert lines[0] == "id\tcategory\tspeed\tsplit\ttext\tterms"
assert len({l.count("\t") for l in lines}) == 1 and lines[0].count("\t") == 5, "tab count"
rows = [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:]]
assert len(rows) == 230, len(rows)
assert len({r["id"] for r in rows}) == 230
cat = collections.Counter(r["category"] for r in rows)
assert cat == {"dev": 150, "general": 50, "trap": 30}, cat
assert {r["speed"] for r in rows} <= {"normal", "fast"} and {r["split"] for r in rows} <= {"tune", "holdout"}
sp = collections.Counter(r["split"] for r in rows)
assert sp == {"tune": 150, "holdout": 80}, sp
for r in rows:
    pre = {"dev": "dev", "general": "gen", "trap": "trap"}[r["category"]]
    assert re.fullmatch(pre + r"\d{3}", r["id"]), r["id"]
    assert r["text"].strip() and r["category"] != "trap" or r["terms"] == ""
    assert r["category"] == "dev" or r["terms"] == "", r["id"]
used = set()
for r in rows:
    if r["category"] == "dev":
        ts = r["terms"].split(";")
        assert 1 <= len(ts) <= 3, (r["id"], ts)
        for t in ts:
            assert t in r["text"], (r["id"], t)
            used.add(t)
vl = open(V, encoding="utf-8").read().rstrip("\n").split("\n")
assert vl[0] == "canonical\taliases"
canon = [l.split("\t")[0] for l in vl[1:]]
assert len(canon) == len(set(canon)) and all(1 <= len(l.split("\t")[1].split(";")) <= 4 for l in vl[1:])
assert used <= set(canon) and set(canon) <= used, (used ^ set(canon))
longn = sum(len(r["text"].replace(" ", "")) >= 50 for r in rows if r["category"] == "dev")
assert longn >= 20, longn
fast = sum(r["speed"] == "fast" for r in rows)
assert 40 <= fast <= 52, fast
for c in cat:
    print(c, collections.Counter((r["split"], r["speed"]) for r in rows if r["category"] == c))
print("rows", len(rows), "category", dict(cat), "split", dict(sp), "fast", fast)
print("terms", len(canon), "long(>=50 chars) dev", longn)
print("OK")
