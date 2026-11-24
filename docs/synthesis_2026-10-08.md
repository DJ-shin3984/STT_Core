# 종합 — Claude × Codex 교차검증 결과 (2026-10-08)

대상: `plan_2026-10-07.md` (1단계 개정 기획안). 검토 1: `review_claude_2026-10-08.md` (Claude Fable 5.1, 웹 1차 출처 열람). 검토 2: `review_codex_2026-10-08.md` (codex GPT-6-Astra xhigh, Exa 검색 + GitHub API 대조). 이 문서는 두 검토를 대조하고, 서로의 오류를 1차 출처로 재확인한 뒤 최종 권고를 적는다.

## 1. 한 줄 결론

기획안의 구조는 맞다. 바꿀 것은 **모델 후보표, 런타임 전략, 게이트 정의, 지연 예산 표** 네 가지다. 가장 큰 변화는 "sherpa-onnx 단일 런타임(1차 Nemotron 스트리밍 + 2차 Qwen3-ASR 오프라인, 둘 다 hotword)"을 기본 스택 후보로 추가하는 것이다. Verbatim 모드는 LLM 없이 구성한다.

## 2. 두 검토가 일치한 것 (그대로 반영)

| 항목 | 내용 |
|---|---|
| 구조 유지 | Controller / Core / Adapter 3계층, 이벤트 발행, Partial은 오버레이 전용, 클립보드 입력 기본값, 1-0 평가 우선 |
| Verbatim에서 LLM 기본 생략 | 단축키 A는 Fast 보정 + 규칙으로 끝낸다. LLM ON/OFF 비교 실험은 1회 허용(codex 조건). LLM은 Polished(단축키 B) 전용 |
| 보정기 구현 축소 | 7개 주체 설계는 유지하되 첫 구현은 "정확 별칭 치환 → 자모 편집거리 후보 → 임계값" 함수 3개. 단, 원문 보존·애매하면 미수정·구간 경계 확인은 처음부터(codex 조건) |
| 1-2 게이트 측정 대상 | 커서에 들어가는 최종 경로 기준으로 용어 재현율을 잰다. 오버레이·STT 원문·Fast 후·LLM 후는 보조 지표 |
| 지연 예산 표 | "STT Final 수신" 한 행을 1차 Final과 2차 재인식으로 분리. 2차는 녹음 종료 즉시 병렬 시작 가능 |
| 부록으로 이동 | 원 기획 검토 표 12행, 2단계 "바꿀 방향" 3항 |
| 로그 | 보존 기간·용량 상한 명시. 디스크 기록은 출력 임계 경로 밖에서 |
| 평가 시점 통일 | 총평·테스트셋 절의 "1-1"을 로드맵과 같은 "1-0"으로 |

## 3. Claude 검토의 오류 — codex가 지적, Claude가 1차 출처로 재확인

| Claude 주장 | 사실 | 확인 방법 |
|---|---|---|
| Qwen3-ASR 공개 API에 context 파라미터 없음 | **있다.** `transcribe(..., context="")`가 시스템 프롬프트로 전달. 스트리밍 상태 생성에도 context 인자 | `qwen_asr/inference/qwen3_asr.py` 100~461행 직접 열람 |
| Nemotron hotword는 sherpa-onnx v1.13.6 이후 | **미포함.** PR #3895 머지 2026-09-14, 최신 릴리스 v1.13.8은 2026-09-10 | GitHub API `pulls/3895` merged_at, `releases` |
| Cohere 10초 재인식 ≈ 50ms, Qwen 300~600ms | **추정.** RTFx 525/148은 A100 배치 측정값 | 리더보드 논문 §3.3 |
| Gemma 4 E4B Q4_K_M 5.07GB | Q4_K_M 4.98GB, Q4_1 5.07GB | codex 파일 목록 확인 |
| EXAONE 개인 사용 무방 | 연구·교육 NC 조건. 업무 도구 용도는 회색지대 → **제외** | 라이선스 원문 |
| 후보 누락 | Fun-ASR-MLT-Nano-2512(800M, 한국어, `hotwords`, Apache 2.0), BuzzASR/korean(한국어 전용 Whisper FT, 2026-09), Kanana-2-3B(2026-07-27), VibeASR.cpp(MIT) | HF 카드·GitHub 확인 |

## 4. Codex 검토의 한계 — Claude 반론

| Codex 주장 | Claude 입장 | 근거 |
|---|---|---|
| "1-0 게이트 '엔진 2종'은 제공 파일에 없어 불일치 확정 불가" | **불일치는 실재한다.** 로드맵 다이어그램 원문에 "엔진 2종 기준선 측정 완료"가 있고 지시서는 3종이다 | Claude가 위젯 코드를 직접 읽음. `plan_2026-10-07.md`에 다이어그램 텍스트 복원 완료 |
| "회사 PC 사양·디스크 735GB는 증빙 없음" | **실측값이다.** CPU 155H 16C/22T, Arc iGPU, RAM 31.5GB, 디스크 여유 735GB, Python 3.12 | PowerShell `Get-CimInstance`로 세션 중 측정 |
| "2주 실사용은 집 전용이 아니다" | **조건부 동의.** 1-5 게이트(수정 없이 쓴 비율 80%)는 최종 채택 구성으로 재야 한다. 최종 구성이 GPU를 쓰면 집, 구성 A0(CPU)면 회사에서도 가능 | 기획안 1-5 게이트 정의 |
| Codex 자체 제약 | 사용량 한도(14:10 리셋)와 샌드박스 쓰기 거부로 파일 저장 실패. 본문은 세션 기록에서 복구했고 내용은 수정하지 않음 | `review_codex_2026-10-08.md` 상단 메모 |

## 5. 양쪽 다 처음엔 놓쳤고 재검증으로 확인된 핵심

**sherpa-onnx에 Qwen3-ASR 오프라인 인식기와 hotwords가 이미 구현돼 있다.** CHANGELOG: Qwen3-ASR 지원(#3399), hotwords(#3434, #3468, #3477), 언어 힌트(#3472), 동음이의어 치환기(#3729), hotword 설정 시 무음 환각 수정(#3907). Codex가 "누락"으로 짧게 언급했고 Claude가 CHANGELOG로 확인했다.

의미:
- 1차(Nemotron 스트리밍)와 2차(Qwen3-ASR 오프라인)를 **같은 런타임, 같은 hotword 파일 형식**으로 돌릴 수 있다.
- Windows 네이티브, 프로세스 내장, CPU↔CUDA 전환이 provider 설정 하나다. 기획안의 "외부 서버 없음·프로세스 내장" 원칙을 vLLM 없이 지킨다.
- `qwen-asr` 0.0.6이 `transformers==4.57.6`을 고정하고 Cohere는 `transformers>=5.4`를 요구하는 **의존성 충돌을 제품 경로에서 제거**한다. transformers/vLLM/OpenVINO 백엔드는 1-0 비교용 별도 가상환경에서만 쓴다.
- 회사 PC에서 2차 STT까지 CPU로 기능 검증이 가능하다(지연은 미측정).

확인되지 않은 것: ONNX int8 변환판의 한국어 CER이 원 모델(Fleurs-ko 2.57)과 같은지, hotword가 한글 발음 표기와 영문 표기 중 어느 쪽에 잘 듣는지. 둘 다 1-0 측정 항목이다.

## 6. 최종 권고 — 기획안 수정 목록

1. **모델 선정 표 교체 (2026-10-08 기준)**
   - 1차 스트리밍: Nemotron 3.5 ASR Streaming 0.6B (sherpa-onnx int8, 160ms부터 시험, 한국어 FLEURS CER 7.1~7.7, OpenMDW-1.1). 대조군: sherpa-onnx zipformer-ko 2024-06-16.
   - 2차 오프라인: Qwen3-ASR-1.7B (sherpa-onnx ONNX 우선, transformers 대조, context/hotword ON·OFF), Cohere Transcribe 03-2026(바이어싱 공식 경로 없음, 한영 혼용·무음 환각 별도 집계), Fun-ASR-MLT-Nano-2512(추가 비교). Qwen3-ASR-0.6B는 속도 대조군.
   - 관찰만: VibeVoice-ASR-Streaming-1.5B(2.9초 청크, MIT), Voxtral Realtime 4B(BF16 16GB 권장), BuzzASR/korean(한국어 전용).
   - LLM: Gemma 4 E4B(Apache 2.0, 임베딩 포함 8B, Q4_K_M 4.98GB), Qwen3.5-4B(Apache 2.0, 2026-03-02, `enable_thinking=False`), 한국어 추가 비교 Kanana-2-3B-instruct. EXAONE 제외.
2. **런타임 원칙 추가**: "STT는 sherpa-onnx 단일 런타임을 기본으로 한다. 다른 백엔드는 비교용 별도 환경." 모델 교체는 경로뿐 아니라 어댑터 종류·샘플레이트·청크·hotword 형식을 함께 설정한다.
3. **1-0 비교 구성 고정**: A0 = Nemotron 단일 + 종료 flush(재인식 없음, LLM 없음) · A1 = A0 + 동일 엔진 재인식 · B = Nemotron 1차 + Qwen3-ASR 2차(context OFF/ON) · C = Nemotron 1차 + Cohere 2차 · D = B/C 부족 시 Fun-ASR-MLT-Nano. 측정은 batch 1, 발화 1/3/10/30초, p50/p95, 후보 하나씩 적재. 1-0 게이트 문구를 "엔진 2종"에서 "구성 A0·B·C 기준표 완성"으로.
4. **Verbatim = LLM 없음**을 기본값으로 명시. 1-4는 Polished 전용 게이트: "등록 용어 보존 100%, 숫자·부정어 변경 0건(시험셋 기준), 종료→입력 p50 700ms".
5. **오보정 게이트 재정의**: 함정 30문장은 "오보정 0건", 일반 50문장은 "오삽입 0건"의 합격/불합격으로. "1% 이하"는 분모(비용어 구간 수)를 정의하고 1-5 실사용 로그에서 측정. 230문장을 사전 조정용 150 / 홀드아웃 80으로 나누고 홀드아웃은 사전·임계값 조정에 쓰지 않는다.
6. **지연 예산 표 재구성**: 1차 Final(청크 크기 + 디코딩) / 2차 재인식(녹음 종료 즉시 병렬) / Fast 20ms / LLM(Polished만, 상한 800ms) / 클립보드 50ms. 버튼 OFF 기준 전체 deadline, 만료 결과 폐기, 발화 ID당 입력 1회.
7. **버퍼 2종 명시**: 발화 전체 버퍼(PTT 상한, 예: 60초)와 2단계용 30초 링버퍼를 구분.
8. **클립보드 입력 보강**: 시퀀스 번호 확인 후 복원, 포커스 변경 시 입력 취소.
9. **VRAM 표 갱신**: Nemotron(CPU 또는 <1GB) + Qwen3-ASR-1.7B(ONNX int8 ≈ 2GB 추정 / bf16 ≈ 3.5GB) + LLM Q4 5~6GB + 캐시·활성값. 합계는 "추정 10~13GB, 동시 적재 peak를 1-0에서 실측".
10. **환경 명시**: Python 3.12. sherpa-onnx는 PR #3895 포함 커밋 고정(또는 포함 릴리스 확인 후). 집 PC는 Blackwell용 CUDA 12.8+ PyTorch Windows wheel. Gemma 4 GGUF는 llama.cpp 최신 빌드.
11. **부록 이동**: 원 기획 검토 표, 2단계 "바꿀 방향".

## 7. 회사 PC / 집 PC 분담 (양측 합의)

| 단계 | 회사 PC (155H, Arc iGPU, 31.5GB) | 집 PC (5080 16GB) |
|---|---|---|
| 1-0 | 골격, 230문장, 녹음 도구, 채점·지연 계측기, 홀드아웃 설계, CPU 기준표(Nemotron·zipformer·Qwen3-ASR ONNX·SenseVoice·Fun-ASR), 의존성 호환성 확인 | VRAM 동시 적재 peak, GPU batch 1 지연 |
| 1-1 | 전부. CPU 첫 글자 지연 측정(목표 달성 여부는 미측정) | GPU 재측정 |
| 1-2 | 전부(hotword/context ON·OFF, 토큰화 검증) | — |
| 1-3 | 전부. Verbatim 기본 경로 완성 | — |
| 1-4 | 기능·정확도(llama.cpp Vulkan/SYCL 또는 CPU, 추정 10~20 tok/s) | 700ms 게이트 |
| 1-5 | 트레이·단축키·클립보드·로그·패키징. 구성 A0면 실사용도 가능 | 최종 구성 2주 실사용 |

## 8. 실측으로만 풀리는 질문 (1-0 완료 기준에 포함)

1. Qwen3-ASR context가 한국어 발화 속 영문 용어 재현율을 얼마나 올리는가. ONNX판과 transformers판이 같은가.
2. Nemotron 160ms 청크의 155H CPU 첫 글자 지연이 300ms 안에 드는가.
3. 2차 재인식 batch 1 지연(5080 / 155H), 10초 발화 기준.
4. Gemma 4 E4B vs Qwen3.5-4B vs Kanana-2-3B의 Polished 품질(의미 보존·용어 보존·지연).
5. 동시 적재 peak VRAM.

## 출처 (추가분, 2026-10-08 열람)

- Qwen3-ASR 소스: https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py
- qwen-asr 의존성: https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml (transformers==4.57.6, vllm==0.14.0)
- sherpa-onnx PR #3895: https://github.com/k2-fsa/sherpa-onnx/pull/3895 (merged 2026-09-14)
- sherpa-onnx CHANGELOG(Qwen3-ASR, hotwords): https://github.com/k2-fsa/sherpa-onnx/blob/master/CHANGELOG.md
- Fun-ASR-MLT-Nano-2512: https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512
- BuzzASR/korean: https://huggingface.co/BuzzASR/korean
- Kanana-2-3B-instruct: https://huggingface.co/kakaocorp/kanana-2-3b-instruct
- VibeASR.cpp: https://github.com/microsoft/VibeASR.cpp
- 나머지 출처는 두 검토서 말미 참조.
