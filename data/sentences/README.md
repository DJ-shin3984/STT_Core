# sentences_v1.tsv

열: `id`(dev001/gen001/trap001), `category`(dev/general/trap), `speed`(normal/fast, fast는 빠르게 읽기), `split`(tune: 사전·임계값 조정용 / holdout: 조정에 쓰지 않는 검증용), `text`(정답 문장), `terms`(dev만, text에 그대로 들어 있는 정식 용어를 ;로 나열).
trap은 용어처럼 들리는 일반 한국어라 terms가 비어 있고, 30문장이 서로 다른 용어(MariaDB, Docker, Redis 등)를 노린다. 용어 기준표는 `data/vocab/terms_v1.tsv`(채점용, 바이어싱용 아님).

| 구분 | 합계 | tune | holdout | fast |
|---|---|---|---|---|
| dev | 150 | 98 | 52 | 38 |
| general | 50 | 33 | 17 | 6 |
| trap | 30 | 19 | 11 | 4 |
| 전체 | 230 | 150 | 80 | 48 |

dev 중 10초 이상(공백 제외 50자 이상) 문장은 23개. 검사: `python -m tests.test_sentences`
