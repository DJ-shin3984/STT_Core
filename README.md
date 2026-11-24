# 실시간 음성 입력 엔진 — 1단계

기획안: `docs/plan_2026-10-07.md` (원문) · `docs/synthesis_2026-10-08.md` (개정 권고). 지금 단계는 **1-0 평가 기반**이다.

## 구조

```
voicecore/            엔진 (Core). Shell 없이 단독 실행·테스트 가능해야 한다
  adapters/base.py    STT 어댑터 인터페이스 (OfflineSTT, StreamingSTT) — 고정
  adapters/registry.py config.toml 섹션 → 어댑터 생성
  adapters/*.py       sherpa_online, sherpa_offline_qwen3asr, qwen3asr_transformers, cohere_transformers
evaltools/            1-0 평가 도구: 녹음, 채점, 지연 계측, VRAM 측정
data/sentences/       테스트 문장 (tsv)
data/recordings/      본인 목소리 녹음 wav + manifest.tsv
data/results/         채점 결과
data/vocab/           용어 사전 (초기엔 비움)
models/               다운로드 모델 (gitignore)
config.toml           모델·장치·단축키 설정
```

## 환경

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

회사 PC(CPU, Intel Arc)는 `provider = "cpu"`. 집 PC(RTX 5080)는 `provider = "cuda"`.
transformers 계열 비교 백엔드는 의존성 충돌 때문에 **별도 venv**에서만 설치한다(`requirements.txt` 주석 참조).

## 1-0 작업 순서

1. `evaltools/record.py` — 문장을 한 줄씩 보여 주고 녹음해 `data/recordings/`에 저장
2. `voicecore/adapters/download.py` — 모델 다운로드
3. `evaltools/run_eval.py` — 녹음 전체를 어댑터로 인식해 `data/results/*.jsonl` 생성
4. `evaltools/score.py` — CER · 용어 재현율 · 오보정 · 지연 p50/p95 표 출력
5. `evaltools/stream_eval.py` — 스트리밍 어댑터의 첫 글자 지연 · 종료 flush 지연
6. `evaltools/vram.py` — 모델 단독/동시 적재 메모리 (GPU는 집 PC)

완료 기준: 구성 A0 · B · C 기준표(CER, 용어 재현율, 오보정, 10초 발화 재인식 지연 p50)가 나온다.

녹음 없이 파이프라인만 확인하려면 모델 동봉 한국어 wav 4개로: `python -m evaltools.run_eval --manifest data/recordings/manifest_smoke.tsv --config-name B`.

## 실행 (1-1 ~ 1-5)

```
python -m shell                                   # 오버레이 + 전역 단축키 (ctrl+alt+a / ctrl+alt+b)
python -m voicecore --wav x.wav --realtime --no-paste   # 파일로 전체 파이프라인 (Shell 없이)
python -m voicecore --mic --seconds 5 --no-paste        # 마이크 N초 (단축키 없이)
```

현황과 실측값은 `docs/progress_2026-10-08.md` (1-0 상세는 `docs/1-0_progress_2026-10-08.md`).
