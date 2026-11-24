"""python -m tests.test_score  -- 합성 JSONL로 CER·재현율·오보정·오삽입·지연 수치 검사."""
import json
import tempfile
from pathlib import Path

from evaltools import score

SENT = """id\tcategory\tspeed\tsplit\ttext\tterms
d1\tdev\tnormal\ttune\tMariaDB와 Spring Boot를 쓴다\tMariaDB;Spring Boot
d2\tdev\tnormal\ttune\tMariaDB 접속\tMariaDB
d3\tdev\tnormal\ttune\tSpring Boot 설정\tSpring Boot
t1\ttrap\tnormal\tholdout\t마리아 접속\t
t2\ttrap\tnormal\tholdout\t봄 부츠 신었다\t
g1\tgeneral\tnormal\tholdout\t오늘 날씨가 좋다\t
g2\tgeneral\tfast\tholdout\t내일 비가 온다\t
"""
TERMS = "canonical\taliases\nMariaDB\t마리아디비\nSpring Boot\t스프링부트;스프링 부트\nRedis\t레디스\n"


def row(id, ref, hyp, ms, sec, biasing="none", config="A0"):
    return dict(id=id, ref=ref, hyp=hyp, elapsed_ms=ms, audio_sec=sec, adapter="fake", model="m",
                provider="cpu", config=config, biasing=biasing, meta={})


def main():
    # 정규화 / Levenshtein
    assert score.norm("Hello, World!  ") == "helloworld"
    assert score.norm("한 글.") == "한글"
    assert score.lev("kitten", "sitting") == 3 and score.lev("", "abc") == 3 and score.lev("abc", "abc") == 0

    rows = [
        row("d1", "MariaDB와 Spring Boot를 쓴다", "mariadb 와 spring boot 를 쓴다.", 100, 2),  # 완전 일치(정규화 후)
        row("d2", "MariaDB 접속", "마리아디비 접속", 200, 2.5),    # alias로 인정 -> 용어 성공, CER > 0
        row("d3", "Spring Boot 설정", "스프링 부츠 설정", 300, 5),  # alias 아님 -> 용어 실패
        row("t1", "마리아 접속", "MariaDB 접속", 150, 4),           # 오보정 1건
        row("t2", "봄 부츠 신었다", "봄 부츠 신었다", 150, 4),       # 정상
        row("g1", "오늘 날씨가 좋다", "오늘 Redis 날씨가 좋다", 400, 12),  # 오삽입 1건
        row("g2", "내일 비가 온다", "내일 비가 온다", 500, 12),
        # 다른 그룹(biasing=hotwords): 한 건만
        row("d1", "MariaDB와 Spring Boot를 쓴다", "MariaDB와 스프링 부트를 쓴다", 1000, 2, biasing="hotwords", config="B"),
    ]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "s.tsv").write_text(SENT, encoding="utf-8")
        (td / "t.tsv").write_text(TERMS, encoding="utf-8")
        j = td / "r.jsonl"
        j.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        gs = score.score_rows(score.load_jsonl(j), score.load_sentences(td / "s.tsv"), score.load_terms(td / "t.tsv"))
        md = score.run(j, td / "s.tsv", td / "t.tsv")
        hold = score.score_rows(score.load_jsonl(j), score.load_sentences(td / "s.tsv"), score.load_terms(td / "t.tsv"), "holdout")

    assert [(g["config"], g["biasing"]) for g in gs] == [("A0", "none"), ("B", "hotwords")], gs
    a, b = gs
    assert a["n"] == 7
    # CER: 편집거리 합 / ref 길이 합 (정규화 후 글자)
    refs = ["mariadb와springboot를쓴다", "mariadb접속", "springboot설정", "마리아접속", "봄부츠신었다", "오늘날씨가좋다", "내일비가온다"]
    hyps = ["mariadb와springboot를쓴다", "마리아디비접속", "스프링부츠설정", "mariadb접속", "봄부츠신었다", "오늘redis날씨가좋다", "내일비가온다"]
    exp_cer = sum(score.lev(h, r) for h, r in zip(hyps, refs)) / sum(map(len, refs))
    assert abs(a["cer"] - exp_cer) < 1e-12, (a["cer"], exp_cer)
    # 용어 재현율: d1 2/2, d2 1/1(alias), d3 0/1 -> 3/4
    assert (a["hit"], a["tot"]) == (3, 4), a
    assert (b["hit"], b["tot"]) == (2, 2), b            # '스프링 부트' alias + MariaDB
    # 오보정: t1 = 1건 / 함정 2문장, 오삽입: g1 = 1건 / 일반 2문장
    assert (a["trap_fp"], a["trap_n"], a["gen_fp"], a["gen_n"]) == (1, 2, 1, 2), a
    # 지연
    assert a["p50"] == 200 and abs(score.pct([100, 200, 300, 150, 150, 400, 500], 95) - a["p95"]) < 1e-9
    assert a["p50_s"] == 150          # <=3s: d1(100) d2(200) -> 150
    assert a["p50_m"] == 150          # 3~10s: d3(300) t1(150) t2(150)
    assert a["p50_l"] == 450          # >10s: 400, 500
    # holdout 필터: A0 그룹에서 t1 t2 g1 g2 만
    assert [g["n"] for g in hold] == [4], hold
    assert "| A0 |" in md and "3/4 (75.0%)" in md and "1/2" in md
    print("test_score OK")
    print(md)


if __name__ == "__main__":
    main()
