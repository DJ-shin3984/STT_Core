"""결과 JSONL -> 지표표(markdown).

python -m evaltools.score data/results/x.jsonl --sentences data/sentences/sentences_v1.tsv --terms data/vocab/terms_v1.tsv
"""
import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


def norm(s: str) -> str:
    """NFC, 공백/문장부호/제어문자 제거, 소문자화."""
    s = unicodedata.normalize("NFC", s).lower()
    return "".join(c for c in s if not (c.isspace() or unicodedata.category(c)[0] in "PZC"))


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * p / 100
    lo = int(k)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (k - lo)


def read_tsv(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        sys.exit(f"파일이 없습니다: {p}")
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def split_semi(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def load_sentences(path) -> dict[str, dict]:
    return {r["id"]: r for r in read_tsv(path)}


def load_terms(path) -> dict[str, list[str]]:
    return {r["canonical"].strip(): split_semi(r.get("aliases")) for r in read_tsv(path) if r["canonical"].strip()}


def load_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        sys.exit(f"결과 파일이 없습니다: {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_rows(rows, sents, terms, split=None) -> list[dict]:
    """(config, adapter, model, biasing) 그룹별 지표 dict 목록."""
    groups = defaultdict(list)
    for r in rows:
        if split and sents.get(r["id"], {}).get("split") != split:
            continue
        groups[(r["config"], r["adapter"], r["model"], r["biasing"])].append(r)
    out = []
    for key, rs in groups.items():
        g = dict(zip(("config", "adapter", "model", "biasing"), key), n=len(rs))
        g["provider"] = "/".join(sorted({r.get("provider", "") for r in rs}))
        edits = reflen = hit = tot = 0
        fp = {"trap": 0, "general": 0}
        cnt = {"trap": 0, "general": 0}
        for r in rs:
            ref, hyp = norm(r["ref"]), norm(r["hyp"])
            edits += lev(hyp, ref)
            reflen += len(ref)
            s = sents.get(r["id"])
            if not s:
                continue
            for t in (t for t in split_semi(s.get("terms")) if norm(t) in ref):
                tot += 1
                hit += any(norm(f) and norm(f) in hyp for f in [t] + terms.get(t, []))
            if s["category"] in fp:
                cnt[s["category"]] += 1
                # 전체 용어 중 ref에 없는데 hyp에 나타난 것 (정식 표기 기준)
                fp[s["category"]] += sum(1 for t in terms if norm(t) not in ref and norm(t) in hyp)
        g.update(cer=edits / reflen if reflen else 0.0, hit=hit, tot=tot,
                 trap_fp=fp["trap"], trap_n=cnt["trap"], gen_fp=fp["general"], gen_n=cnt["general"])
        ms = [r["elapsed_ms"] for r in rs]
        g.update(p50=pct(ms, 50), p95=pct(ms, 95))
        for name, lo, hi in (("s", -1, 3), ("m", 3, 10), ("l", 10, 1e9)):
            g["p50_" + name] = pct([r["elapsed_ms"] for r in rs if lo < r["audio_sec"] <= hi], 50)
        out.append(g)
    return sorted(out, key=lambda g: (g["config"], g["adapter"], g["model"], g["biasing"]))


def table(groups: list[dict]) -> str:
    f = lambda v: "-" if v is None else f"{v:.0f}"
    head = ["config", "adapter", "model", "provider", "biasing", "n", "CER%", "용어 재현율",
            "오보정(함정)", "오삽입(일반)", "p50ms", "p95ms", "p50 <=3s", "p50 3~10s", "p50 >10s"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in groups:
        rec = f"{g['hit']}/{g['tot']} ({100 * g['hit'] / g['tot']:.1f}%)" if g["tot"] else "-"
        lines.append("| " + " | ".join(map(str, [
            g["config"], g["adapter"], g["model"], g["provider"], g["biasing"], g["n"], f"{100 * g['cer']:.2f}", rec,
            f"{g['trap_fp']}/{g['trap_n']}", f"{g['gen_fp']}/{g['gen_n']}",
            f(g["p50"]), f(g["p95"]), f(g["p50_s"]), f(g["p50_m"]), f(g["p50_l"])])) + " |")
    return "\n".join(lines)


def run(jsonl, sentences, terms, split=None) -> str:
    return table(score_rows(load_jsonl(jsonl), load_sentences(sentences), load_terms(terms), split))


def main(argv=None):
    ap = argparse.ArgumentParser(description="결과 JSONL을 CER·용어 재현율·오보정·지연 표로 요약한다.")
    ap.add_argument("jsonl", nargs="+", help="결과 JSONL (여러 개면 합쳐서 그룹별로 한 행씩)")
    ap.add_argument("--sentences", default="data/sentences/sentences_v1.tsv")
    ap.add_argument("--terms", default="data/vocab/terms_v1.tsv")
    ap.add_argument("--split", choices=["tune", "holdout"], help="해당 split 문장만 채점")
    a = ap.parse_args(argv)
    rows = [r for p in a.jsonl for r in load_jsonl(p)]
    print(table(score_rows(rows, load_sentences(a.sentences), load_terms(a.terms), a.split)))


if __name__ == "__main__":
    main()
