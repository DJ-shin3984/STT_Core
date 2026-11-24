# 1단계 기획안 검토 — Claude (2026-10-08 기준 최신 정보)

> **정정 (2026-10-08, codex 교차검증 반영 후 Claude가 1차 출처로 재확인)**
>
> 1. **Qwen3-ASR 컨텍스트 API — 내가 틀렸다.** `qwen_asr/inference/qwen3_asr.py`의 `transcribe(..., context="")`가 시스템 프롬프트로 전달된다(README에는 미문서화). 아래 본문의 "공개 API에 파라미터 없음", "2차 후보 어느 쪽도 바이어싱 보장 안 됨"은 철회한다. 효과량은 1-0에서 context ON/OFF로 실측한다.
> 2. **Nemotron hotword 버전 — 내가 틀렸다.** sherpa-onnx PR #3895는 2026-09-14 머지됐고, 최신 릴리스 v1.13.8(2026-09-10)에 포함되지 않았다. "v1.13.6 이후"는 오류. 해당 커밋을 포함한 소스 빌드 또는 다음 릴리스가 필요하다.
> 3. **RTFx → 지연 환산 수치는 추정이다.** Cohere "≈50ms", Qwen "300~600ms", 구성 A/C "400ms 이내"는 A100 배치 처리량에서 유도한 추정값이며 채택 근거로 쓸 수 없다. 같은 PC에서 batch 1로 실측한다.
> 4. **Gemma 4 E4B GGUF**: Q4_K_M은 4.98GB, 5.07GB는 Q4_1 파일이다.
> 5. **누락 후보 추가**: Fun-ASR-MLT-Nano-2512(800M, 한국어 포함 31개 언어, `hotwords` 인자, Apache 2.0), BuzzASR/korean(Whisper-large-v3 한국어 전용 미세조정, 2026-09, 코드스위칭 미검증), Kanana-2-3B-instruct(2026-07-27, Kanana Open License), VibeASR.cpp(MIT, CPU). EXAONE "개인 사용 무방" 표현은 철회하고 후보에서 제외한다.
> 6. **가장 큰 추가 발견**: sherpa-onnx에 Qwen3-ASR **오프라인 구현과 hotwords**가 이미 있다(CHANGELOG #3399, #3434, #3468, #3477, #3729, #3907). Windows 네이티브·프로세스 내장·CPU/CUDA 전환이 가능하고, `qwen-asr` 0.0.6의 `transformers==4.57.6` 고정과 Cohere의 `transformers>=5.4` 충돌도 피한다. 종합 문서 `synthesis_2026-10-08.md` 참조.


검토 대상: `docs/plan_2026-10-07.md` (실시간 음성 입력 엔진 — 기획 검토 및 1단계 개정 기획안, 2026-10-07판)
검토 조건: 무료·로컬 실행 범위에서 속도와 정확도 최우선. 집 PC = RTX 5080 16GB / 9950X / 128GB. 회사 PC = Core Ultra 7 155H / Intel Arc iGPU / 31.5GB RAM / NVIDIA 없음.

## 결론 요약

1. **구조는 그대로 간다.** Controller/Core/Adapter 3계층, 바이어싱을 1차 방어선으로, Partial은 오버레이에만, 1-0 평가 기반 선행. 모두 2026-10 기준으로도 옳다.
2. **모델 후보는 갱신이 필요하다.** 1차 STT로 적은 sherpa-onnx 한국어 zipformer는 2024-06 모델이다. 2026-06 출시 Nemotron 3.5 ASR Streaming 0.6B가 sherpa-onnx에서 한국어 스트리밍 + hotword(2026-08-24 머지)를 지원하므로 이것이 1차 후보가 돼야 한다.
3. **기획안의 가장 큰 가정 오류: Qwen3-ASR-1.7B의 "문맥 텍스트 바이어싱".** 기술보고서(arXiv 2601.21337)는 "시스템 프롬프트 안의 컨텍스트 토큰"을 언급하지만, 공개된 `qwen-asr` 패키지 README·모델 카드에는 context/hotword 파라미터가 없다(2026-10-08 확인). Cohere Transcribe는 바이어싱 미지원이 모델 카드로 확인된다. 즉 **커서에 들어가는 2차 STT 결과에는 현재 후보 어느 쪽도 바이어싱이 보장되지 않는다.** 이 모순을 1-0에서 풀어야 한다.
4. **Verbatim 모드에서 LLM을 빼는 것이 지연 예산을 가장 크게 아낀다.** 오인식·용어·문장부호 수정은 Fast 보정 + 규칙으로 충분하고, 기획안 13번 안전검사가 LLM 변경의 대부분을 되돌린다. LLM은 Polished(단축키 B)에만 쓴다.
5. **회사 PC에서 1-0, 1-1, 1-2, 1-3 전부와 1-5 대부분을 개발·측정할 수 있다.** 집 PC가 꼭 필요한 것은 VRAM 실측, GPU 지연 게이트(1-4), 2주 실사용뿐이다.

---

## 1. 퍼포먼스 — 무료·로컬 범위에서의 모델 후보 갱신

### 1-1. 1차 STT (말하는 중, 오버레이용 스트리밍)

| 후보 | 출시 | 라이선스 | 한국어 | 용어 바이어싱 | 실행 환경 | 판정 |
|---|---|---|---|---|---|---|
| sherpa-onnx streaming zipformer ko (2024-06-16) — 기획안 현안 | 2024-06 | Apache 2.0 | KsponSpeech 학습 | hotwords O (bpe 단위, modified_beam_search) | CPU, int8 121MB | 대조군으로 유지 |
| **Nemotron 3.5 ASR Streaming 0.6B** (nvidia) | 2026-06-04 | OpenMDW-1.1 | ko-KR "transcription-ready", FLEURS ko CER 7.1~7.6 | hotwords O — sherpa-onnx PR #3895 (2026-08-24), v1.13.6 이후 | CPU int8 627MB, 청크 80/160/560/1120ms, beam=4 CPU RTF≈0.14 | **1차 권고** |
| Qwen3-ASR-0.6B / 1.7B 스트리밍 모드 | 2026-01-29 | Apache 2.0 | Fleurs-ko CER 3.72 / 2.57 | 공개 API에 없음 | **vLLM 전용**(Windows 비공식, WSL2 또는 커뮤니티 포크), 2초 청크 | 1차용 부적합 |
| Voxtral Realtime 4B (Mistral) | 2026-02-04 | Apache 2.0 | 13개 언어 포함 | Mini Transcribe V2(API)만 명시 | vLLM, 4B bf16 ≈ 8GB | VRAM 과다, 제외 |
| VibeVoice-ASR-Streaming-1.5B (Microsoft) | 2026-09-02 | MIT | 10개 언어 포함 | hotword O (`--context_info`) | 연구용 소스 설치, 실제 3B/5.6GB, 2.9초 청크 + 0.5초 lookahead | 청크가 커서 1차용 부적합. 2차 후보로 관찰 |

권고: 1차 = Nemotron 3.5 Streaming (sherpa-onnx, 160ms 청크부터 시험), 대조군 = zipformer-ko. 둘 다 CPU에서 돌아가므로 1-1·1-2 전부 회사 PC에서 가능하다.

### 1-2. 2차 STT (발화 종료 후 전체 재인식, 커서 입력용)

| 후보 | 구조 | 한국어 수치 | 바이어싱 | 속도 | Windows 실행 |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | 오디오 인코더 + LLM 디코더 | Fleurs-ko CER 2.57, CommonVoice-ko 5.88, MLC-SLM-ko 8.61 | **미확인** (보고서 언급, 공개 API 없음) | 리더보드 RTFx 148. LLM 디코딩이라 10초 발화 재인식에 수백 ms 예상(미실측) | transformers 백엔드 O. OpenVINO 2026.3+ CPU/GPU O |
| Qwen3-ASR-0.6B | 동일 | Fleurs-ko 3.72, CV-ko 8.48 | 동일 | 1.7B보다 빠름 | 동일 |
| Cohere Transcribe 03-2026 (2B) | Conformer 인코더 + 경량 Transformer 디코더 | 14개 언어 포함, 한국어 수치는 그래프로만 공개 | **없음** (모델 카드 미언급) | RTFx 525 → 10초 발화 수십 ms | transformers O |
| Nemotron 3.5 1120ms 청크를 2차로 재사용 | FastConformer-RNNT | CER ≈ 7.1 | hotwords O | CPU RTF 0.14 | sherpa-onnx O |
| VibeVoice-ASR-Streaming-1.5B | Qwen2 1.5B 백본 | 수치 미공개(이미지만) | hotword O | 미공개 | 연구용 설치 |

핵심 쟁점: 커서에 들어가는 텍스트는 2차 결과인데, 정확도가 가장 높은 Qwen3-ASR은 바이어싱이 불확실하고 Cohere는 아예 없다. 바이어싱이 확실한 쪽(Nemotron, VibeVoice)은 CER이 높거나 미검증이다. 따라서 **1-0에서 아래 3가지 구성을 같은 테스트셋으로 비교하고, "용어 재현율 + CER + 10초 발화 재인식 지연"으로 고른다.**

- 구성 A: Nemotron 단일 엔진 (1차 = 2차, hotword 적용, 발화 종료 시 링버퍼 재디코딩). 가장 단순, CPU만으로 동작, 회사 PC에서도 실사용 가능.
- 구성 B: Nemotron 1차 + Qwen3-ASR-1.7B 2차 (바이어싱 없음, 사후 보정 의존).
- 구성 C: Nemotron 1차 + Cohere 2차 (바이어싱 없음, 가장 빠름).

Qwen3-ASR의 바이어싱은 "가능"이 아니라 1-0 실험 항목으로 바꾼다. 채팅 템플릿의 시스템 프롬프트에 용어 목록을 넣었을 때 용어 재현율이 오르는지 측정해 기록한다.

### 1-3. LLM (Final Corrector)

| 모델 | 확인 사항 |
|---|---|
| Gemma 4 E4B | 2026-03-31 출시, **Apache 2.0**(Gemma 라이선스에서 전환). 실제 파라미터는 임베딩 포함 8B. Q4_K_M GGUF 5.07GB. RTX 5080 추정 ≈ 95 tok/s |
| Qwen3.5-4B | 2026-02~03 출시, Apache 2.0. 4B급 한국어 품질은 Gemma 4 E4B와 비교해 테스트셋으로 결정 |
| EXAONE 계열 | 비상업 라이선스. 개인 사용은 무방하나 기본 후보에서 제외 |

기획안 VRAM 표의 "LLM 4B급 4bit 3~6GB"는 Gemma 4 E4B 기준 **5~7GB**로 상향한다. 합계는 Nemotron(<1GB 또는 CPU) + Qwen3-ASR-1.7B bf16 ≈ 4GB + LLM 5~7GB + Windows 1GB ≈ 11~13GB로 16GB 안에 든다.

### 1-4. 지연 예산 재계산

| 구간 | 기획안 | 검토 의견 |
|---|---|---|
| 마지막 청크 + STT Final | 200~300ms | 1차(Nemotron)는 청크 크기(160~560ms)가 곧 지연. 2차가 Qwen3-ASR-1.7B면 LLM 디코딩이라 10초 발화에 300~600ms 추정 → 단독으로 예산 초과 위험. Cohere면 ≈ 50ms. 이 행을 "1차 Final"과 "2차 재인식"으로 분리해야 한다 |
| Final LLM | 300ms (상한 800) | 4B Q4 ≈ 95 tok/s → 30토큰 출력 ≈ 320ms + 프리필. Verbatim에서는 생략, Polished에서만 |
| 합계 (Verbatim) | 700ms | LLM 생략 + 구성 A 또는 C면 ≈ 400ms 이내 가능. 구성 B면 700ms 경계 |

---

## 2. 불필요하거나 줄일 내용

| 항목 | 판정 | 이유 |
|---|---|---|
| "원 기획 검토" 표 12행 | 부록으로 이동 | 이력 기록이다. 1-0 착수에 필요 없다 |
| "2단계 방향성 검토 > 바꿀 방향" 3항 | 2단계 설계 때로 미룸 | "1단계에서 미리 심어둘 것" 4항만 1단계 문서에 남긴다 |
| Correction Engine 주체 7개 | 설계는 유지, **구현 순서 명시** | 1-3 첫 구현은 Dictionary 정확일치 + 자모 편집거리 + 임계값 1개(함수 3개)로 시작. Span Detector·Context Scorer·2단계 임계값은 1-3 게이트(오보정 1%) 실패 시에만 추가 |
| STT 표 "참고: VibeVoice-ASR-Streaming-7B" | 1.5B로 교체 | 7B는 제외 사유가 명확하고, 1.5B가 실제 후보다 |
| "모델 실행 방식: 외부 서버 프로그램 안 씀" | 문장 수정 | Qwen3-ASR 스트리밍은 vLLM 엔진이 필요하다(Windows 비공식). 오프라인(2차) 용도만 transformers로 프로세스 내장 가능 |
| Stable 구간 + 말하는 중 Fast 보정 | 유지, "오버레이 전용" 명시 | 2차 전체 재인식 구조에서 이 보정은 커서 입력에 영향이 없다 |
| 1-0 지시서 4번 "STT 어댑터 3종" vs 1-0 게이트 "엔진 2종" | 숫자 통일 | 아래 3절 참조 |

---

## 3. 설계 정확성과 단계 구분

맞는 것: 3계층 책임 분리, 이벤트 발행 인터페이스, 용어 목록의 2곳 주입, Partial 오버레이 전용, 클립보드 입력 기본값, 1-0 평가 기반 선행, 수치 게이트, 30초 링버퍼.

고칠 것:

1. **1-0 게이트 ↔ 지시서 불일치.** 게이트는 "엔진 2종 기준선", 지시서는 "STT 어댑터 3종". 위 구성 A/B/C 기준으로 "스트리밍 1종(Nemotron, 대조군 zipformer-ko) + 2차 2종(Qwen3-ASR, Cohere)"으로 통일.
2. **1-2 게이트(용어 재현율 85%)의 측정 대상이 불명확.** 커서에 들어가는 것은 2차 결과이므로 게이트는 2차 출력 기준이어야 한다. 그런데 2차 후보는 바이어싱이 없거나 미확인이다. 수정: 1-2 게이트 = "커서 입력 경로 기준 용어 재현율 85%". 1차 바이어싱은 오버레이 품질 보조 지표로 분리.
3. **1-4 재정의.** Verbatim은 LLM 없이 1-3에서 완성되므로, 1-4는 Polished 전용 게이트로 바꾼다. 게이트: "Polished 출력에서 등록 용어 보존 100%, 숫자·부정어 변경 0건, 종료→입력 p50 700ms".
4. **1-1 게이트와 청크 크기 연결.** Nemotron 560ms 청크면 청크 자체가 첫 글자 지연 하한이다. "첫 글자 p50 300ms" 게이트는 160ms 청크 기준으로 명시.
5. **지연 예산 표에 2차 재인식 행 추가.** 현재 "STT Final 수신" 한 행에 1차와 2차가 뭉쳐 있다.
6. **VRAM 표 갱신.** 1-3절 참조.
7. **수정 로그의 디스크 상한.** 발화 오디오를 전부 저장하므로 보존 기간 또는 용량 상한 한 줄이 필요하다.
8. **환경 명시.** Python 3.12, sherpa-onnx ≥ v1.13.6(현재 v1.13.8, 2026-09-10), 집 PC는 Blackwell용 CUDA 12.8+ PyTorch. Gemma 4 GGUF는 llama.cpp 최신 빌드에서 토크나이저 이슈가 있었으므로(2026년 상반기 이슈) 최신 릴리스 사용.

단계 구분 자체(1-0 → 1-5, 각 게이트)는 올바르다. 위 1~5번을 반영하면 단계가 "측정 가능한 단위"로 정확해진다.

---

## 4. 회사 PC에서 구현 가능한 범위

회사 PC 사양(실측): Intel Core Ultra 7 155H(16코어/22스레드, NPU 내장), Intel Arc iGPU(공유 메모리), RAM 31.5GB, Python 3.12, 디스크 여유 735GB, NVIDIA 없음 → CUDA 스택 불가.

| 단계 | 회사 PC에서 | 집 PC가 필요한 것 |
|---|---|---|
| 1-0 평가 기반 | 골격, 테스트 문장 230개, 녹음 도구, 채점 스크립트, 지연 계측기 **전부**. STT 기준선: Nemotron·zipformer-ko·SenseVoice-Small(sherpa-onnx CPU), Qwen3-ASR(OpenVINO 2026.3+ CPU/iGPU 또는 transformers CPU, 느림), Cohere(transformers CPU, Conformer라 수 초 내) | VRAM 실측, GPU 기준 재인식 지연 |
| 1-1 스트리밍 입력 | **전부**. Nemotron CPU RTF 0.14이므로 첫 글자 지연 게이트도 CPU에서 측정 가능 | 없음 |
| 1-2 용어 사전 + 바이어싱 | **전부** (sherpa-onnx hotword는 CPU) | Qwen3-ASR 시스템 프롬프트 실험은 GPU가 편함 |
| 1-3 Fast 보정 | **전부** (순수 CPU 코드) | 없음 |
| 1-4 Final 보정(LLM) | 기능 개발·정확성 테스트 가능. llama.cpp Vulkan/SYCL(Arc iGPU) 또는 CPU, 4B Q4 ≈ 10~20 tok/s 추정 → 문장당 2~3초 | 지연 게이트(700ms) |
| 1-5 제품화 | 전역 단축키, 클립보드 입력, 트레이, 수정 로그 **전부** | 2주 실사용 |

부수 효과: 구성 A(Nemotron 단일)가 채택되면 회사 PC에서도 Verbatim 모드를 CPU만으로 실사용할 수 있다. whisper.cpp v1.9.5(2026-10-06)는 Vulkan으로 Arc iGPU를 쓰지만, 155H에서 실시간 배수가 1배 안팎이라 이 프로젝트의 1차 후보는 아니다.

---

## 5. 1-0 첫 작업 지시서 수정안

- 4번 교체: "STT 어댑터를 같은 인터페이스로 만든다. 스트리밍: sherpa-onnx Nemotron 3.5 ASR Streaming 0.6B(160ms·560ms), 대조군 zipformer-ko. 2차: Qwen3-ASR-1.7B(transformers), Cohere Transcribe. 구성 A/B/C를 모두 측정한다."
- 추가: "Qwen3-ASR에 용어 목록을 시스템 프롬프트로 넣는 실험을 하고, 용어 재현율 변화를 기록한다. 효과가 없으면 비교표에 '바이어싱 불가'로 적는다."
- 추가: "Verbatim 경로는 LLM 없이 구성한다. LLM은 Polished 전용."
- 완료 기준에 추가: "10초 발화 기준 2차 재인식 지연 p50, 구성 A/B/C별."
- 환경: Python 3.12, sherpa-onnx ≥ 1.13.6, 집 PC CUDA 12.8+.

---

## 출처 (2026-10-08 열람)

- Qwen3-ASR GitHub README: https://github.com/QwenLM/Qwen3-ASR (스트리밍은 vLLM 전용, context 파라미터 없음)
- Qwen3-ASR 기술보고서: https://arxiv.org/html/2601.21337v1 (Fleurs-ko CER 2.57/3.72, 2초 청크, 시스템 프롬프트 컨텍스트 언급)
- Cohere Transcribe 모델 카드: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 (Apache 2.0, Conformer+Transformer, RTFx 525, 바이어싱 미언급)
- Nemotron 3.5 ASR Streaming 모델 카드: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b (OpenMDW-1.1, FLEURS ko CER 7.12~8.31, 청크 80~1120ms)
- sherpa-onnx Nemotron 지원 문서: https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html
- sherpa-onnx hotword for Nemotron (PR #3895, 2026-08-24): https://github.com/k2-fsa/sherpa-onnx/issues/3572
- sherpa-onnx 한국어 zipformer hotword: https://github.com/k2-fsa/sherpa-onnx/issues/1762
- VibeVoice-ASR-Streaming-1.5B 분석: https://www.orcarouter.ai/blog/vibevoice-asr-streaming-1-5b-explained
- Voxtral Transcribe 2 / Realtime: https://mistral.ai/news/voxtral-transcribe-2/
- Open ASR Leaderboard 논문 표: https://arxiv.org/html/2510.06961v4
- Gemma 4 E4B GGUF: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF (8B with embeddings, Q4_K_M 5.07GB)
- OpenVINO 릴리스 노트 (Qwen3-ASR CPU/GPU 2026.3+): https://docs.openvino.ai/releasenotes
- vLLM Windows 현황: https://fazm.ai/t/vllm-windows-support-2026
- whisper.cpp 1.8.3 iGPU(Core Ultra 7 155H): https://www.phoronix.com/news/Whisper-cpp-1.8.3-12x-Perf
- Phonon-2(영어 전용, 제외): https://www.fermionresearch.com/models/phonon-2/
