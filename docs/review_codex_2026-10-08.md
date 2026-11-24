<!-- 복구 메모: codex(GPT-6-Astra xhigh) 워커가 2026-10-08 작성. codex 샌드박스가 파일 쓰기를 거부해 Claude가 codex 세션 기록(apply_patch 호출 본문)에서 그대로 복구해 저장함. 내용은 수정하지 않음. -->

# 실시간 음성 인식 엔진 1단계 기획안 교차검증 — Codex

검토 기준일: **2026-10-08**  
대상: [기획안 원문](plan_2026-10-07.md), [Claude 검토서](review_claude_2026-10-08.md)  
조건: 무료로 로컬 실행, 속도·한국어 정확도 우선. 회사 PC는 Core Ultra 7 155H / Intel Arc iGPU / RAM 31.5GB / Python 3.12, 집 PC는 Ryzen 9950X / RTX 5080 16GB / RAM 128GB.

검증 방법: Exa Search로 API·스트리밍·LLM·누락 모델·실행 환경의 5개 축을 조사하고, 모델 개발자의 카드·소스·논문과 GitHub PR/API·릴리스를 직접 대조했다. 아래 성능 수치는 출처의 측정값이며 **두 PC에서 모델 추론을 실측한 결과가 아니다.** 하드웨어 성능 예상은 **추정**, 확인할 수 없는 사항은 **미확인**, 설계 제안은 **권고**로 구분한다. 문서에 없는 내용을 추정해 원문 사실로 취급하지 않았다.

## 1. 먼저 바로잡아야 할 결론

1. **반대 — Qwen3-ASR의 공개 컨텍스트 API는 있다.** 공식 `qwen-asr` 구현의 `transcribe(context=...)`가 시스템 프롬프트로 연결된다. Transformers 공식 문서도 `prompt`를 통한 context/hotword 사용을 설명한다. 따라서 Claude의 “2차 후보 모두 바이어싱을 보장할 수 없다”는 논거 중 Qwen 부분은 잘못됐다. 기능 지원과 한국어 용어 정확도 개선량은 별개다. [공식 구현](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_asr.py), [Transformers 문서](https://huggingface.co/docs/transformers/main/model_doc/qwen3_asr)
2. **반대 — Nemotron hotword의 머지 날짜·최소 버전이 틀렸다.** PR #3895는 **2026-09-14 UTC 머지**이며, 8월 24일 머지가 아니다. 최신 정식 릴리스로 확인한 **v1.13.8(9월 10일)**에도 해당 커밋이 없다. `sherpa-onnx>=1.13.6` 설치만으로 기능이 제공되지 않는다. [PR](https://github.com/k2-fsa/sherpa-onnx/pull/3895), [v1.13.8](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8), [커밋 비교](https://github.com/k2-fsa/sherpa-onnx/compare/v1.13.8...3df7adacc0d5f140e4e81b746dac0fbeeac16fab)
3. **반대 — 처리량으로 개인 PC의 응답 시간을 단정할 수 없다.** Cohere RTFx 525, Qwen RTFx 148은 A100 80GB의 배치 평가다. “10초 재인식 약 50ms”, “구성 C가 가장 빠름”, “CPU RTF 0.14이므로 첫 글자 목표 달성”은 이 프로젝트에서 입증되지 않았다. [리더보드 논문 §3.3](https://arxiv.org/html/2510.06961v4), [Nemotron PR 작성자의 측정](https://github.com/k2-fsa/sherpa-onnx/pull/3895)
4. **부분 동의 — Verbatim은 LLM 생략을 기본값으로 비교할 가치가 크다.** 하지만 “규칙만으로 충분”, “안전검사가 LLM 변경 대부분을 되돌림”은 측정 근거가 없다. LLM의 정확도 개선량과 지연을 따로 비교해야 한다.
5. **부분 동의 — 회사 PC에서 대부분의 기능 개발은 가능하다.** 그러나 GPU 성능 게이트·전체 조합의 메모리 검증까지 완료한 것과는 다르다. 반대로 2주 실사용은 집 PC 전용 작업이 아니다.

## 2. Qwen3-ASR-1.7B / 0.6B 검증

### 2.1 공개 API에 context/hotword가 없다는 주장: 반대

공식 패키지 소스에는 다음 경로가 실제로 있다.

- `Qwen3ASRModel.transcribe(..., context="", language=...)`
- `_build_messages()`가 context를 `role="system"` 메시지에 넣는다.
- `_build_text_prompt()`가 `processor.apply_chat_template()`를 호출한다.
- Transformers 백엔드도 해당 컨텍스트를 사용한다. vLLM 전용 기능이 아니다.
- 스트리밍 상태 생성에도 `init_streaming_state(context=...)`가 있다.

[공식 소스](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_asr.py)

예를 들어 패키지 모델 객체를 초기화한 뒤 다음처럼 호출하는 경로가 공개되어 있다. 아래는 API 사용 예시이며, 이 검토에서 실행하지는 않았다.

```python
result = model.transcribe(
    audio="sample.wav",
    context="개발 용어: MariaDB, Spring Boot, 커넥션 풀",
    language="Korean",
)
```

PyPI 공개판은 0.0.6이며 저장소의 패키지 메타데이터도 같은 버전을 명시한다. README의 간단한 예제가 context를 생략한 것을 API 부재로 판단한 것이 Claude의 오류다. [PyPI](https://pypi.org/project/qwen-asr/0.0.6/), [패키지 메타데이터](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml)

별도의 최신 Transformers 네이티브 구현에서는 `processor.apply_transcription_request(audio=..., prompt=..., language=...)`를 제공한다. 공식 문서에 **Context / hotwords** 절과 시스템 메시지를 사용하는 채팅 템플릿 예제가 있다. 두 구현의 클래스·체크포인트·버전 요구사항은 구분해야 한다. [Transformers 공식 문서](https://huggingface.co/docs/transformers/main/model_doc/qwen3_asr)

**정정:** “Qwen은 컨텍스트 기반 용어 유도가 가능하다. 별도의 단어별 점수 부스팅 API와는 방식이 다르며, 한국어·영문 혼용 효과는 실측한다.”

“효과가 없으면 바이어싱 불가로 기록”이라는 Claude 지시도 **반대**한다. 특정 사전·프롬프트·테스트셋에서 효과가 없는 것은 기능 부재가 아니다. “지원 O / 해당 조건 개선 없음”으로 기록해야 한다.

### 2.2 한국어 정확도 수치: 동의

공식 기술보고서 부록의 한국어 CER(%)는 다음과 같다.

| 모델 | FLEURS-ko | CommonVoice-ko | MLC-SLM-ko |
|---|---:|---:|---:|
| Qwen3-ASR-0.6B | 3.72 | 8.48 | 10.31 |
| Qwen3-ASR-1.7B | 2.57 | 5.88 | 8.61 |

Claude가 인용한 수치는 맞다. 다만 개발용어 혼용 발화, ONNX/int8 변환판, 다른 평가자의 정규화 설정에서 같은 CER을 보장하지 않는다. [기술보고서 부록 A](https://arxiv.org/html/2601.21337v1)

### 2.3 “vLLM 전용 스트리밍, 2초이므로 1차 부적합”: 부분 동의

공식 `qwen-asr` 패키지의 스트리밍 구현은 vLLM 백엔드만 지원하며 기본 `chunk_size_sec=2.0`이다. 첫 Partial 300ms 목표의 기본 후보로 우선순위가 낮다는 판단은 타당하다. 그러나 2초는 API 기본값이고 조절할 수 있으므로 고정 하한이라고 단정해서는 안 된다. 해당 구현은 매 청크 누적 오디오를 다시 입력하므로, 캐시를 활용하는 RNNT의 스트리밍 비용과도 다르다. [공식 구현](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_asr.py)

또한 **vLLM 백엔드 사용과 외부 서버 운영은 같은 뜻이 아니다.** 공식 패키지는 `Qwen3ASRModel.LLM(...)` 형태의 Python 사용도 제공한다. Windows 네이티브 지원 문제와 프로세스 내장 여부는 별도 판단이다. 원문에서 이미 정한 단일 프로그램 원칙은 유지하면서 가능한 백엔드를 선정하면 된다. [Qwen README](https://github.com/QwenLM/Qwen3-ASR), [vLLM 설치 문서](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)

**누락:** Qwen3-ASR의 sherpa-onnx 오프라인 구현에도 hotwords 지원 이력이 있다(PR #3434, #3468, #3477). 따라서 회사 PC 후보가 Transformers/OpenVINO뿐이라는 인상도 바로잡아야 한다. ONNX판의 정확도와 프롬프트 처리 일치 여부는 별도 확인한다. [sherpa-onnx 변경 기록](https://github.com/k2-fsa/sherpa-onnx/blob/master/CHANGELOG.md)

## 3. Nemotron 3.5 ASR Streaming 0.6B 검증

### 3.1 한국어·라이선스·출시: 동의

NVIDIA 카드에 2026-06-04 공개, 약 600M 파라미터, 한국어 `ko-KR`의 transcription-ready 지원, OpenMDW-1.1 라이선스가 명시되어 있다. 영어 전용 이전 모델 `nemotron-speech-streaming-en-0.6b`와 구분해야 한다. [공식 카드](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

OpenMDW-1.1은 무료 사용·변경·배포를 허용하는 허용적 라이선스이며, 배포 시 라이선스와 원 저작권·출처 고지를 유지하는 조건이 있다. Apache-2.0이나 비상업 라이선스와 혼동하면 안 된다. [라이선스 원문](https://openmdw.ai/license/1-1/), [공식 설명](https://openmdw.ai/about/)

### 3.2 한국어 CER 7.1~7.6: 부분 동의

카드의 한국어 결과는 CER이며 조건별 차이가 있다.

| 설정 | 80ms | 160ms | 320ms | 560ms | 1120ms |
|---|---:|---:|---:|---:|---:|
| 언어 지정 | 7.59 | 7.70 | 7.27 | 7.18 | 7.12 |
| 언어 자동 감지 | 8.31 | 8.18 | 7.81 | 7.49 | 7.30 |

Claude의 좁은 범위는 권고한 160ms 설정의 7.70까지 빠뜨린다. 원 모델 카드 수치이지 sherpa-onnx int8 + 한국어 hotword 조합의 검증 결과도 아니다. 언어 지정/자동 감지, 청크, 양자화, 디코더를 함께 기록해야 한다. [공식 평가표](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b#performance)

### 3.3 “PR #3895가 8월 24일 머지, v1.13.6 이상 지원”: 반대

직접 확인한 이력은 다음과 같다.

| 항목 | 확인 결과 |
|---|---|
| PR 생성 | 2026-08-25 00:33:43 UTC |
| PR 머지 | **2026-09-14 02:25:43 UTC** |
| 머지 커밋 | `3df7adacc0d5f140e4e81b746dac0fbeeac16fab` |
| v1.13.6 | 2026-08-18 공개 — 해당 PR 이전 |
| v1.13.8 | 2026-09-10 공개 — 해당 PR 이전 |
| 릴리스 API의 최신 정식판 | v1.13.8 |
| v1.13.8 → PR 머지 커밋 비교 | 머지 커밋 쪽이 4커밋 앞섬. v1.13.8에 미포함 |

[PR 원문](https://github.com/k2-fsa/sherpa-onnx/pull/3895), [PR API](https://api.github.com/repos/k2-fsa/sherpa-onnx/pulls/3895), [v1.13.6](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.6), [최신 릴리스 API](https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/latest), [커밋 비교](https://github.com/k2-fsa/sherpa-onnx/compare/v1.13.8...3df7adacc0d5f140e4e81b746dac0fbeeac16fab)

**정정:** 소스에는 streaming NeMo transducer의 `modified_beam_search`와 hotwords 지원이 머지되어 있다. 그러나 이 검증에서 확인한 정식 릴리스에는 아직 없다. 사용하려면 **해당 커밋을 포함하는 소스/검증된 빌드를 고정**해야 한다. 확인되지 않은 최초 지원 버전 번호를 만들어 적어서는 안 된다.

일반 단어 형태의 hotword를 넣을 때는 BPE 설정·어휘와 실제 토큰화를 확인해야 한다. PR에 기능이 있다는 사실만으로 한국어 개발 용어 부스팅 효과까지 검증된 것은 아니다. [PR 구현·사용 조건](https://github.com/k2-fsa/sherpa-onnx/pull/3895)

### 3.4 “CPU int8 627MB, RTF 0.14이므로 1차 최우선”: 부분 동의

- **크기 정정:** sherpa 문서의 627M은 encoder 파일이다. decoder 약 14M, joiner 약 9.1M도 있으므로 전체 모델이나 실행 RAM이 627MB인 것은 아니다. [공식 ONNX 모델 목록](https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html)
- **속도 정정:** RTF 약 0.14는 PR 작성자가 **1120ms int8, CPU, beam=4**로 측정한 사례다. 155H, 160ms, Python 바인딩, 한국어 발화의 실측값이 아니다. 160ms 청크에서는 호출 빈도와 캐시 처리 비용도 달라진다. [PR 측정 조건](https://github.com/k2-fsa/sherpa-onnx/pull/3895)
- **선정 권고:** Nemotron은 중요한 비교 후보지만 자동 우승자는 아니다. 회사 PC에서는 설치가 확인된 한국어 Zipformer를 기준선으로 먼저 측정하고, Nemotron의 추가 정확도·지연·빌드 비용을 비교하는 것이 합리적이다.

## 4. Cohere Transcribe 03-2026 검증

### 4.1 2B·한국어·Apache-2.0·Transformers 지원: 동의

Conformer 계열 인코더와 경량 Transformer 디코더를 쓰는 약 2B 모델이고, 한국어를 포함한 14개 언어와 Apache-2.0을 명시한다. 카드의 현재 설치 예시는 Transformers 5.4.0 이상이다. [공식 카드](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)

### 4.2 “바이어싱 미지원이 모델 카드로 확인”: 부분 동의

기본 후보표에서 “공식 contextual-biasing 인터페이스 미확인”으로 취급하는 것은 타당하다. 다만 카드가 기능을 언급하지 않는다는 사실만으로 **명시적으로 불가능하다고 선언했다**고 쓸 수는 없다.

공식 Transformers 프로세서는 언어·문장부호 옵션으로 디코더 프롬프트를 구성하며, 검토한 공개 인터페이스에는 Qwen 같은 자유 텍스트 context/hotword 인자가 없다. 이 근거로 **공식 지원 경로 없음/미확인**이라고 표현하는 것이 정확하다. 임의 디코더 수정까지 원리적으로 불가능하다는 뜻은 아니다. [프로세서 구현](https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere_asr/processing_cohere_asr.py)

### 4.3 실시간 스트리밍: 부분 동의, 구분 보완 필요

공개 사용법은 전체 오디오 또는 분할한 파일을 처리한다. **캐시를 유지하며 새 마이크 입력을 이어 받는 네이티브 스트리밍 경로는 확인하지 못했다.** vLLM 온라인 서빙, 생성 텍스트의 스트리밍 전송, 데이터셋 로딩의 `streaming=True`는 입력 음성의 네이티브 스트리밍과 다르다. 현재 기획에서는 2차 오프라인 재인식 후보로 분류하는 것이 맞다. [공식 사용법](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026), [개발진 기술 설명](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)

### 4.4 한국어 수치와 “리더보드 1위”: 부분 동의

- **5.42%는 영어 평균 WER**이며 한국어 CER이 아니다. 1위라는 설명도 **2026-03-26 영어 평가**의 결과다. 10월의 한국어 최상위 모델이라는 근거가 되지 않는다.
- 공식 카드·기술 글의 한국어 비교는 언어별 그래프로 제시되며, 한국어는 CER을 사용한다고 명시한다. 해당 그래프는 FLEURS·Common Voice 17.0 등을 언어별로 적용한 평균이다.
- **한국어의 정확한 숫자는 이번 검증에서 미확인.** 텍스트 표에 값이 없고 그래프 원본 접근도 실패했으므로 숫자를 추정해 넣지 않았다. “그래프 중심으로 공개”라는 Claude의 관찰에는 동의하지만, 이를 한국어 정확도 우열의 근거로 사용할 수는 없다.

[모델 카드 결과](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026#results), [개발진 기술 글의 Figure 4 설명](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release)

**Claude가 빠뜨린 중요한 제약:** 개발진은 단일 언어 지정을 전제로 학습했으며 코드 스위칭 성능이 일관되지 않고, 무음·비음성에서 환각이 생길 수 있다고 설명한다. 한국어+영문 개발 용어가 핵심인 이 프로젝트에서는 반드시 혼용·무음 테스트가 필요하다. [공식 제한사항](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026#limitations)

### 4.5 RTFx 525 → 10초 수십 ms/약 50ms: 반대

RTFx 525 자체와 Qwen의 148은 리더보드 논문에서 확인된다. 그러나 측정 환경은 **A100-SXM4-80GB, 가능한 경우 batch 64**이다. 전체 음성 길이/배치 처리 시간으로 계산한 처리량을 batch 1의 발화 종료 후 지연으로 바꾸면 안 된다. 10/525초라는 산술값도 약 19ms일 뿐, 개인 PC에서 보장되는 응답 시간이 아니다. [평가 환경·표 3](https://arxiv.org/html/2510.06961v4)

따라서 Claude의 Cohere 약 50ms, Qwen 300~600ms, 구성 A/C 400ms 이내는 모두 **추정이며 채택 근거로는 부족**하다. 같은 PC에서 같은 길이의 발화를 batch 1로 재야 한다.

## 5. 나머지 STT 후보에 대한 판정

| Claude 주장 | 판정 | 독립 검증·보완 |
|---|---|---|
| 한국어 Zipformer 2024-06-16, CPU 대조군 유지 | **동의** | 공식 문서가 해당 체크포인트와 한국어 전용 범위를 확인한다. 다만 121M은 encoder만이며 decoder/joiner도 필요하다. 오래됐다는 이유만으로 낮은 우선순위를 확정할 수 없다. [모델 문서](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html) |
| Zipformer hotword는 BPE + modified_beam_search | **부분 동의** | 해당 디코딩 방식의 지원은 맞다. 그러나 한국어 전용 모델이 MariaDB 같은 라틴 표기를 원하는 대로 생성할지는 토큰화와 학습 범위에 좌우된다. 한글 발음 후보→정식 표기 후처리가 여전히 필요할 수 있다. [hotwords 문서](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html) |
| VibeVoice-ASR-Streaming-7B 대신 1.5B를 검토 | **부분 동의** | **둘 다 실제 공개 모델**이다. 7B를 잘못된 모델명으로 취급할 이유는 없다. 1.5B는 LM 백본 크기이며 카드의 전체 모델 표시는 약 3B다. 한국어·hotword·MIT 지원은 확인된다. [1.5B 카드](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B), [7B 카드](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) |
| VibeVoice 2.9초 청크 + 0.5초 lookahead이므로 첫 Partial용 부적합 | **동의** | 공식 보고서가 공개 체크포인트의 해당 구성을 확인한다. 300ms 목표와 맞지 않는다. 다만 Claude의 Qwen2 백본 표기는 **Qwen2.5**로 정정한다. 9월 2일은 논문 공개일, GitHub 공지는 9월 3일이다. [논문](https://arxiv.org/html/2609.02812), [공식 README](https://github.com/microsoft/VibeVoice) |
| VibeVoice 한국어 수치는 이미지만 존재 | **부분 동의** | 1.5B의 한국어 수치는 이번 검증에서도 확정하지 않았다. 다만 공식 논문에는 **7B의 MLC-Challenge 한국어 CER 9.09**가 텍스트 표로 있다. 1.5B 수치로 대입해서는 안 된다. [논문 표 1](https://arxiv.org/html/2609.02812) |
| Voxtral Realtime 4B는 한국어·Apache-2.0, 메모리 때문에 제외 | **부분 동의** | 지원·라이선스는 맞다. 공식 카드는 BF16 실행에 GPU 16GB 이상을 안내한다. 단순 “가중치 약 8GB”는 실행 메모리가 아니다. 2차 STT+LLM 동시 상주의 기본 후보에서는 부담이 크지만, LLM 없는 단일 엔진 구성까지 일괄 배제할 이유는 없다. [공식 카드](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) |
| Voxtral의 바이어싱은 Mini Transcribe V2에 명시 | **동의** | 공식 발표에서 context biasing은 해당 모델 기능으로 소개된다. 이를 Realtime 공개 가중치의 hotword 기능으로 옮겨 적지 않는다. [Mistral 발표](https://mistral.ai/news/voxtral-transcribe-2) |
| whisper.cpp v1.9.5는 10월 6일 출시, 155H에서 약 실시간 1배 | **부분 동의** | 버전과 날짜는 맞다. 하지만 릴리스 노트는 155H·모든 Whisper 모델의 RTF≈1을 보장하지 않는다. 인용한 과거 버전의 외부 벤치마크로 현재 turbo/양자화/Vulkan/OpenVINO 조합 전체를 제외하면 과도하다. [v1.9.5 릴리스](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5) |

## 6. LLM 라이선스·크기·VRAM 및 빠진 후보

### 6.1 Gemma 4 E4B / Qwen3.5-4B: 부분 동의

| 항목 | 확인한 사실 | Claude 검토서 정정 |
|---|---|---|
| Gemma 4 E4B 라이선스·날짜 | Apache-2.0. 공식 릴리스 기록은 2026-03-31, 발표 블로그는 04-02 | Claude의 3월 31일은 근거가 있다. 원문의 “4월 출시”와 단순 모순으로 처리하지 않는다. [릴리스 기록](https://ai.google.dev/gemma/docs/releases), [Google 발표](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) |
| E4B 파라미터 | **유효 4.5B, 임베딩 포함 8B**라는 공식 규격 | “4B 가중치 모델”로 계산하면 안 된다. [공식 카드](https://ai.google.dev/gemma/docs/core/model_card_4) |
| E4B 메모리 | Google 표는 BF16 17.9GB, Q4_0 4.5GB. 문맥·런타임에 따른 추가 메모리 별도 | Claude의 5~7GB는 가능한 **추정**이지 고정 VRAM이 아니다. [공식 메모리 표](https://ai.google.dev/gemma/docs/core) |
| E4B GGUF 파일 | 확인한 Unsloth **Q4_K_M은 4.98GB**, **Q4_1은 5.07GB** | “Q4_K_M 5.07GB”는 파일을 혼동했다. 디스크 파일 크기와 실행 VRAM도 구분한다. [Q4_K_M 파일](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q4_K_M.gguf), [파일 목록](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/tree/main) |
| Qwen3.5-4B | Apache-2.0, 언어모델 4B, 2026-03-02 공개 | “2~3월”보다 정확히 쓸 수 있다. [모델 카드](https://huggingface.co/Qwen/Qwen3.5-4B), [공식 출시 기록](https://github.com/QwenLM/Qwen3.5) |
| Qwen 양자화·thinking | 확인한 Q4_K_M 파일 2.74GB. thinking은 기본 켜짐, `enable_thinking=False`로 제어 | `/nothink` 문구만 붙여 끄는 방식은 공식 지원하지 않는다. [GGUF 파일](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/blob/main/Qwen3.5-4B-Q4_K_M.gguf), [공식 사용법](https://huggingface.co/Qwen/Qwen3.5-4B) |
| RTX 5080 약 95 tok/s | 동일 모델·양자화·컨텍스트·백엔드의 1차 출처 실측을 확인하지 못함 | **추정**으로만 남긴다. 30토큰 생성 시간에 프리필·템플릿·GPU 경합·한글 토큰 수가 추가된다. |

**VRAM 결론:** 집 PC 16GB에서 STT 1개+양자화 LLM 1개의 실행은 **가능성이 있는 구성이라는 추정**이다. 1차 STT를 CPU로 돌리고, 짧은 컨텍스트·출력 상한을 사용해 비교할 가치가 크다. 그러나 “합계 11~13GB이므로 반드시 들어감”에는 동의하지 않는다.

Qwen3-ASR 1.7B의 16bit 가중치를 단순 계산하면 약 3.4GB이고, 여기에 LLM 파일만 더해도 런타임 캐시·활성값·워크스페이스·화면 메모리가 빠져 있다. 서로 다른 PyTorch/llama.cpp 할당기는 같은 프로세스라는 이유만으로 메모리를 자동 통합 관리하지 않는다. **동시 적재 후 긴 발화·동시 작업의 peak를 실측**해야 한다. ForcedAligner를 추가하면 별도 모델의 비용도 포함한다.

### 6.2 더 나은 4B급 한국어 LLM이 있는가

**결론: 검토 후보는 더 있지만, 한국어 STT 문장 보정에서 Qwen3.5-4B/Gemma 4 E4B보다 우수하다고 확정할 공통 실험 근거는 확인하지 못했다.**

| 추가 후보 | 확인한 사실 | 이 프로젝트에서의 판정 |
|---|---|---|
| **Kanana-2-3B-Instruct** | 2026-07-27 가중치 공개. 한국어 평가 포함. Kanana Open License | Claude의 누락이다. 3B급 한국어 보정 비교 후보로 적절하다. 공식 비교 대상은 주로 더 작은 Qwen 모델이므로 Qwen3.5-4B/Gemma E4B보다 좋다는 결론은 못 낸다. [카드](https://huggingface.co/kakaocorp/kanana-2-3b-instruct) |
| **EXAONE 4.0 1.2B** | 한국어 지원 소형 모델. 확인한 라이선스는 연구·교육 목적의 NC 조건 | 4B보다 작아 회사 PC의 실험 후보가 될 수 있으나 회사 업무 기본 모델로 자동 채택하지 않는다. Claude의 “개인 사용은 무방”도 사용 목적을 생략한 과장이다. [카드](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B), [라이선스](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/blob/main/LICENSE) |
| HyperCLOVA X SEED 4B | NAVER의 2026-06 공식 발표·평가는 확인 | 해당 **4B 모델의 공개 다운로드 가중치·라이선스를 이번 조사에서 확인하지 못했으므로** 무료 로컬 실행 후보로 확정하지 않는다. 발표와 가중치 공개는 다르다. [NAVER 기술 글](https://clova.ai/en/tech-blog/seeing-and-hearing-the-field-for-ourselves-building-our-own-encoders-and-hyperclova-x-seed-4b) |

Kanana 라이선스는 Apache/MIT와 다르다. 원문 §4에는 API·클라우드 형태의 제3자 제공이나 모델 재배포 사업 등의 별도 상업 조건이 있다. **무료 로컬 내부 사용 후보와 아무 제한 없는 재배포 가능 모델을 구분**한다. [라이선스 원문](https://huggingface.co/kakaocorp/kanana-2-3b-instruct/blob/main/LICENSE)

활성 파라미터가 3~4B인 대형 MoE를 “4B급 VRAM 모델”로 추가해서도 안 된다. 예를 들어 Kanana-2-30B-A3B는 총 30B 모델이다. [공식 카드](https://huggingface.co/kakaocorp/kanana-2-30b-a3b-instruct-2601)

**권고:** 기본 비교는 Qwen3.5-4B와 Gemma 4 E4B, 한국어 특화 추가 비교는 Kanana-2-3B로 제한한다. 일반 지식·수학 점수 대신 의미 보존, 숫자·부정어 보존, 용어 표기, 군말 제거, 잘못된 수정 수, 발화 종료 후 지연으로 고른다.

## 7. Claude가 놓친 한국어 지원 STT 후보

아래는 추가 후보의 존재를 확인한 목록이다. 모두 구현하라는 뜻은 아니며, “최신 최강 모델” 순위도 아니다.

| 후보 | 한국어·라이선스·주요 사실 | 우선순위 판단 |
|---|---|---|
| **Fun-ASR-MLT-Nano-2512** | 약 800M, 한국어 포함 31개 언어, Apache-2.0. 공식 FunASR 호출에 `hotwords`, 언어, ITN 옵션이 있다. **일반 Fun-ASR-Nano-2512와 언어 범위가 다르다.** [카드](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512), [README](https://github.com/FunAudioLLM/Fun-ASR) | **우선 추가 비교.** 한국어+컨텍스트+소형이라는 요구에 직접 대응한다. 표에 있는 중국어/영어 성능을 MLT의 한국어 실력으로 대입하지 않는다. 네이티브 저지연 스트리밍 후보로는 확인하지 못했다. |
| **Whisper large-v3-turbo** | 약 809M의 다국어 모델, MIT. large-v3의 디코더를 줄인 모델. [카드](https://huggingface.co/openai/whisper-large-v3-turbo) | **2차 인식 속도 대조군.** Claude는 whisper.cpp만 주변적으로 언급했고 turbo를 본 비교에서 빠뜨렸다. CPU int8/faster-whisper 또는 whisper.cpp 경로를 실제로 측정한다. [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| **Meta omniASR CTC 300M/1B v2** | v2 체크포인트 공개, Apache-2.0 계열 공개 모델. 공식 언어 목록에 `kor_Hang` 포함. CTC 300M 실제 파라미터는 약 325M이다. [공식 저장소](https://github.com/facebookresearch/omnilingual-asr), [언어 목록](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py) | CPU·오프라인 정확도/속도 보조 대조군. 네이티브 스트리밍·한국어 hotword 경로는 별도 확인 대상. 이름의 “300M”만 보고 LLM 변형까지 같은 크기로 보면 안 된다. |
| **VibeVoice-ASR-BitNet / VibeASR.cpp** | 공식 CPU 구현, 약 1.58GB의 양자화 구성. 한국어 MLC-KO 평가와 `--context` 옵션이 있다. [공식 README](https://github.com/microsoft/VibeASR.cpp), [추론 코드](https://github.com/microsoft/VibeASR.cpp/blob/main/demo/asr_infer.cpp) | 회사 PC 연구 후보. 파일/청크 처리와 토큰 스트리밍을 300ms 네이티브 음성 스트리밍으로 혼동하지 않는다. Windows·155H 성능은 미측정. |
| **BuzzASR/korean** | 2026-09 공개 연구의 한국어 Whisper-large-v3 미세조정판, MIT. 저자 보고 CER: FLEURS 5.16, Common Voice 25 4.24, 합산 4.61. [모델 카드](https://huggingface.co/BuzzASR/korean), [9월 9일 논문](https://arxiv.org/abs/2609.09554) | **최신 한국어 특화 후보로 누락.** 단일 한국어 모델이므로 한영 코드 스위칭을 별도 검증해야 한다. 다른 저자의 Qwen CER과 그대로 우열 비교하지 않는다. |
| SenseVoiceSmall | 한국어 지원·소형 비자기회귀 모델. [공식 카드](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) | Claude가 회사 PC 표에 이미 넣었으므로 “완전 누락”은 아니다. 본 비교표의 가벼운 CPU 기준선으로 승격할 수 있다. |

FireRedASR2의 다국어 **언어 감지** 지원을 한국어 **전사** 지원으로 잘못 읽지 않아야 한다. 공식 ASR 설명의 중심은 중국어·방언·영어다. GLM의 상용 다국어 API 지원 역시 공개 Nano 체크포인트의 보증으로 대체할 수 없다. [FireRed 공식 설명](https://github.com/FireRedTeam/FireRedASR), [GLM Nano 카드](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)

**실행 권고:** 1-0에서 전부 어댑터로 제품화하지 않는다. 먼저 한국어 Zipformer/Nemotron, Qwen3-ASR-0.6B·1.7B, Cohere, Fun-ASR-MLT-Nano의 파일 평가를 하고, 필요할 때 turbo/SenseVoice를 속도 기준선으로 보탠다. 성능 개선이 확인된 경로만 Core에 통합한다.

## 8. 불필요한 내용·설계·단계에 대한 주장별 검토

이 절의 판정은 두 로컬 문서를 직접 비교한 **설계 판단**이다. 외부 구현 사실에는 별도 URL을 붙였다.

| Claude의 주장 | 판정 | 독립 판단 |
|---|---|---|
| Controller/Core/Adapter 분리, 이벤트 발행, Partial 오버레이 전용 유지 | **동의** | 책임 분리와 안정적인 외부 앱 입력에 적절하다. 다만 원문의 아키텍처는 Adapter를 외부 계층으로 설명하면서 처리표에는 커서 입력을 Core 책임으로 넣는다. Core는 결과를 발행하고 출력 Adapter가 Windows 입력을 담당하도록 문구를 통일한다. |
| 바이어싱을 모델 선정의 최우선 조건으로 유지 | **부분 동의** | 바이어싱은 유용한 수단이다. 최우선은 실제 최종 출력의 정확도·용어 재현율·오삽입률·지연이다. 바이어싱이 약해도 기본 정확도가 더 높은 모델을 기계적으로 탈락시켜서는 안 된다. |
| Verbatim에서 LLM 제거 | **부분 동의** | 기본 경로 생략에 찬성한다. 다만 “규칙만으로 모든 오인식·문장부호 해결”, “안전검사로 대부분 취소”는 입증되지 않았다. LLM OFF/ON의 실제 수정 이득과 손해를 비교한다. |
| 원 기획 검토 이력·2단계 세부 설명을 줄임 | **동의** | 착수 본문은 현재 구조·인터페이스·측정·단계에 집중하면 된다. 개인화 학습, KWS, 화자 라우팅 상세는 후속 단계로 미룬다. 원문은 이번 작업에서 수정하지 않는다. |
| 보정기를 처음부터 7개 클래스로 만들 필요 없음 | **동의** | 함수 몇 개와 공통 후보 자료형으로 시작할 수 있다. 사전 자동 발음 생성도 최초에는 수동 검수한 별칭으로 대체 가능하다. |
| Span/Context/도메인 안전장치는 1% 게이트 실패 시에만 추가 | **부분 동의** | 복잡한 점수기는 미뤄도 된다. 그러나 구간 경계 확인·원문 보존·애매하면 미수정 같은 최소 안전장치는 처음부터 있어야 한다. 유사도 임계값 하나가 일반어 오보정을 충분히 막는다는 근거는 없다. |
| Stable 보정은 오버레이 전용 | **동의** | 2차 재인식 결과가 이를 덮어쓴다. 원본 가설과 표시용 보정 가설을 따로 유지한다. “3회 같음”은 휴리스틱이며 향후 수정 불가 확정이 아니다. |
| “STT가 서버 방식과 맞지 않음”을 고침 | **동의** | 세션 유지 WebSocket 서버도 스트리밍할 수 있다. 외부 서버 미사용은 사용자 선택으로 유지하되, 기술적으로 불가능하다는 근거를 삭제하는 것이 맞다. 같은 프로세스도 prefix cache를 자동 보장하지 않는다. [VibeVoice 공식 스트리밍 서버 예제](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr-streaming.md) |
| 모델은 경로 설정만 바꾸면 교체 | **부분 동의** | 같은 어댑터 안의 호환 모델에는 가능하다. Qwen/Cohere/RNNT 간에는 샘플레이트·청크·토크나이저·언어 코드·hotword·캐시·타임스탬프 지원이 다르다. 어댑터 종류와 capability도 설정해야 한다. |
| 1-0의 “엔진 2종 게이트”와 “어댑터 3종” 불일치 | **부분 동의** | 원문 텍스트의 지시서·완료 기준에는 모두 3종이 있다. “2종 게이트”는 제공 파일에 보이지 않는다. 로드맵은 **EMBED 자리표시자**라 확인 불가다. 불일치를 확정하지 말고 실제 단계 표를 문서에 복원해야 한다. |
| 1-2 용어 재현율은 최종 입력 경로 기준 | **동의** | 오버레이, 최종 STT 원문, Fast 이후, LLM 이후를 각각 측정하되 사용자 KPI는 실제 입력본에 적용한다. Qwen 컨텍스트 부재를 이유로 이 단계가 모순이라는 설명은 철회해야 한다. |
| 1-4는 Polished만 다룸 | **부분 동의** | 개발 범위 축소로는 타당하다. 단 Verbatim의 LLM 비교 실험을 금지할 필요는 없다. 숫자·부정어 “0건 변경”은 시험셋 조건이고 의미 보존 전체의 증명이 아니다. Polished와 Verbatim의 지연 목표를 구분한다. |
| 첫 글자 300ms는 Nemotron 160ms 기준 | **부분 동의** | 청크 대기+추론+UI 처리로 예산을 나누는 데 찬성한다. 160ms를 선택했다고 300ms를 달성하는 것은 아니다. 단어가 완성될 때까지 발행하지 않는 지연도 측정해야 한다. |
| 지연 예산에 2차 재인식 행 추가 | **동의** | 반드시 필요하다. 다만 1차 Final을 기다려야만 독립적인 2차 재인식을 시작할 수 있는 것은 아니다. 녹음 종료 오디오가 준비되면 병렬 시작 가능한 의존 관계를 구분한다. |
| 30초 링버퍼는 그대로 맞음 | **부분 동의** | 최근 구간 보관에는 적절하다. **전체 재인식용 발화 저장소를 30초 링버퍼 하나로 대체하면 긴 PTT 발화 앞부분이 사라진다.** 발화 길이 상한 또는 별도 발화 버퍼/청크 저장이 필요하다. |
| 로그 보존 기간·용량 상한 추가 | **동의** | 원문에는 명시되지 않았다. 디스크 기록은 출력의 임계 경로에서 분리한다. 사용자 수정으로 확인된 정답만 정답 로그로 취급한다. |
| 클립보드 기본 입력 유지 | **부분 동의** | 실용적인 시작점이다. 그러나 Ctrl+V 전송 성공이 실제 삽입 완료를 뜻하지는 않는다. 포커스 변화·관리자 권한 앱·비동기 붙여넣기·사용자의 새 클립보드를 덮어쓰는 복원 경쟁을 처리해야 한다. [SendInput 제약](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput), [클립보드 변경 번호](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getclipboardsequencenumber) |

### Claude가 추가로 놓친 구현·평가 문제

1. **Python 의존성 충돌.** 공개 `qwen-asr==0.0.6`은 `transformers==4.57.6`을 고정하지만 Cohere의 현재 네이티브 예시는 `transformers>=5.4.0`을 요구한다. 원문의 단일 Python 프로그램에 둘을 단순 설치할 수 있다고 가정하면 안 된다. 모델 비교용 환경을 분리하고, 최종 제품에는 호환되는 네이티브 Transformers 구현 또는 ONNX/OpenVINO 경로를 선정한다. 무검증 `--no-deps`는 해결책으로 삼지 않는다. [Qwen 의존성](https://github.com/QwenLM/Qwen3-ASR/blob/main/pyproject.toml), [Cohere 설치](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026#usage)
2. **오보정률 1%와 함정 문장 30개의 표본 크기.** 문장 단위 오보정률이면 1건만 틀려도 3.33%다. 독립 시행이라고 가정할 때 30건에서 0건이어도 실제 오류율의 단측 95% 상한은 약 9.5%다. 약 1% 이하라는 통계적 근거를 원하면 0실패 기준으로 약 299개의 독립 부정 사례가 필요하다. 이는 이항모형 계산이며 실제 발화 간 독립성도 고려해야 한다. 분모를 문장/비용어 구간 중 무엇으로 할지 먼저 정의한다.
3. **평가셋 누출.** 같은 230문장에서 틀린 용어를 사전에 추가하고 그 문장만 반복 평가하면 개인화 효과를 과대평가할 수 있다. 사전 작성·임계값 조정용 세트와 표현·문맥이 다른 고정 홀드아웃을 나눈다. 함정 세트뿐 아니라 일반 문장의 불필요한 변경·용어 오삽입도 센다.
4. **CER 정규화와 코드 표기 지표 분리.** 공백·문장부호를 제거한 한국어 CER과 `MariaDB`, 버전 숫자, 변수명, 부정어의 정확한 표기를 함께 평가한다. 사전에 단어가 있다고 정상 문장에 삽입하는 경우까지 잡아야 한다. Polished는 문장을 의도적으로 바꾸므로 CER 하나로 품질을 판정하지 않는다.
5. **단일 엔진의 최단 경로가 빠졌다.** Nemotron을 1차=2차로 사용해도 매번 전체 링버퍼를 다시 디코딩할 필요는 없다. **스트리밍 종료 flush만 하는 경로**를 먼저 기준선으로 두고, 큰 청크 재인식이 실제로 더 좋아질 때 추가한다. RTF 0.14를 그대로 가정해도 10초 전체 재처리는 1.4초이므로 “재디코딩+400ms 이내” 주장은 성립하지 않는다.
6. **deadline·취소·중복 출력.** LLM에만 800ms 상한을 두면 700ms 전체 목표를 보장하지 못한다. 버튼 OFF 기준 전체 deadline, 다음 발화와의 순서, 만료된 작업 결과 폐기, 발화 ID별 단 한 번 입력을 정의한다. 오디오 캡처 콜백에서는 모델 호출·파일 기록을 하지 않고 bounded queue와 초과 처리 정책을 둔다.
7. **원문 내 평가 착수 시점도 혼재한다.** 총평·테스트셋 절에는 Phase 1-1, 후반 로드맵에는 1-0이 나온다. 평가는 1-0으로 통일한다. 실제 아키텍처·단계별 게이트는 EMBED가 아닌 읽을 수 있는 표/그림으로 제공해야 한다.
8. **“수정 없이 쓴 비율”의 관측 방법.** 수정 단축키를 누르지 않았다는 사실만으로 정확한 발화라고 판단할 수 없다. 사용자가 표본을 확인하는 방식 등 분모·관측 기준이 필요하다.

## 9. 회사 PC 범위와 현실적인 단계

### 9.1 환경 주장 검증

| Claude 주장 | 판정 | 검증 |
|---|---|---|
| NVIDIA가 없으므로 CUDA 경로 불가 | **동의** | 회사 PC는 CPU 또는 Intel용 백엔드를 사용한다. 사용자 제공 사양을 전제로 한 판단이며 이 검토에서 하드웨어를 직접 실측하지 않았다. |
| Qwen3-ASR는 OpenVINO 2026.3+ CPU/iGPU에서 가능 | **부분 동의** | 2026.3(8월 4일) 릴리스에 **CPU/GPU early release** 지원이 명시되어 있다. 후보 경로로 타당하지만 Windows·155H·변환 모델·context 전달까지 검증됐다는 뜻은 아니다. NPU 지원으로 확대 해석하지 않는다. [Intel 릴리스 노트](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html) |
| llama.cpp Vulkan/SYCL로 Arc 활용 | **동의** | SYCL 공식 문서는 Intel GPU와 Windows 경로를 설명한다. 단 선택한 최신 Gemma/Qwen 아키텍처·양자화의 지원 조합을 고정해야 한다. [SYCL 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) |
| 4B Q4 10~20 tok/s, 문장당 2~3초 | **부분 동의** | **추정**에 불과하다. 모델·프리필·한글 출력 길이·전력 설정·공유 메모리 대역폭·ASR 경합을 고정한 실측이 필요하다. |
| Cohere는 Conformer이므로 CPU 수 초 이내 | **반대** | 구조 이름만으로 155H의 처리 시간을 알 수 없다. CPU 기능 실험은 가능성이 있지만 시간 상한은 **미확인**이다. |
| 집 PC는 Blackwell 대응 CUDA 12.8+ PyTorch 필요 | **부분 동의** | Blackwell 지원과 CUDA 12.8 wheel 도입은 공식 기록에 있다. “CUDA Toolkit 버전만 맞춤”으로 끝내지 말고 Windows 지원 PyTorch wheel·드라이버·사용 커널을 함께 검증한다. [PyTorch 2.7 발표](https://pytorch.org/blog/pytorch-2-7/) |
| 회사 PC 디스크 여유 735GB는 실측 | **부분 동의** | Claude의 원측정 증빙은 제공된 파일에 없다. 이번 검토에서는 해당 수치를 독립 확인한 사실로 사용하지 않는다. |

RAM 31.5GB는 소형 STT·양자화 LLM 개발을 진행할 현실적인 용량이라는 **추정**이다. 다만 Intel iGPU는 시스템 RAM을 공유하므로 31.5GB를 전용 VRAM처럼 계산하면 안 된다. 모델 파일이 들어가는 것과 실시간 지연 충족은 다르다.

### 9.2 단계별 권고

아래는 읽을 수 있는 형태로 제안하는 **권고 단계**이며, 원문의 보이지 않는 EMBED를 복원했다고 주장하는 표가 아니다.

| 단계 | 회사 PC에서 수행할 일 | 완료 판단 / 집 PC로 넘길 일 |
|---|---|---|
| **1-0 평가·호환성** | 녹음·정답·홀드아웃 설계, 파일 추론 스크립트, CER/용어/오보정/지연 계측, CPU 모델 기준선, 패키지 호환성 확인 | 회사 PC 기준표를 완성한다. GPU VRAM·5080 지연 게이트는 **미완료 항목으로 분리**한다. 후보마다 제품 어댑터를 먼저 만들 필요는 없다. |
| **1-1 최소 PTT** | 단축키 A/B, 캡처, CPU 스트리밍, 오버레이, 종료 flush, 단 한 번 출력 | CPU에서 첫 Partial·종료 지연을 실측한다. 두 단축키를 처음부터 연결한다는 사용자 결정은 유지한다. B는 보정 완성 전 A와 동일하게 동작한다. |
| **1-2 용어 주입** | 실제 선택한 최종 STT 경로의 context/hotword ON/OFF, 사전 버전·별칭·토큰화 검증 | 최종 출력의 용어 재현율과 오삽입을 확인한다. Nemotron hotword는 정식판 설치만으로 완료 처리하지 않는다. |
| **1-3 Fast 보정** | 정확 별칭 치환→보수적 발음 후보 순서, 변경 이력·원문 보존·fallback | 일반 문장 손상 없이 개선되는지 홀드아웃 검증. 회사 PC에서 개발·평가 가능. Verbatim의 기본 경로를 여기서 완성한다. |
| **1-4 선택적 Final** | Qwen/Gemma 등의 짧은 비추론 프롬프트, Polished, 보존 검사, deadline·취소 | 회사 PC는 기능·정확도 평가. 집 PC는 최종 STT+LLM 동시 peak 및 batch 1 p50/p95, 출력 길이별 지연을 검증한다. |
| **1-5 제품화** | 트레이·설정·로그 상한·포커스·클립보드 경쟁·중복 입력 방지·패키징 | **두 PC 모두 실사용 가능.** 집 PC에서는 CUDA 배포, 회사 PC에서는 CPU/iGPU 배포와 실제 업무 앱 호환성을 확인한다. |

“1-0~1-3을 회사 PC에서 전부 완료”라는 말은 **기능 개발 기준으로는 대체로 타당**, 집 PC용 수치 게이트까지 통과한다는 의미로는 **반대**다. 회사/집 프로파일을 나누면 회사 PC 개발을 GPU 검증 때문에 불필요하게 멈출 이유도 없다.

### 9.3 최소 성능 비교 구성

- **A0:** CPU 스트리밍 단일 엔진 → 종료 flush → 최소 Fast → 입력. LLM 없음.
- **A1:** A0에 동일 엔진의 재인식을 추가. 실제 정확도 이득이 있을 때만 유지.
- **B:** 스트리밍 오버레이 + Qwen3-ASR 최종 인식. **context OFF/ON을 둘 다 측정.**
- **C:** 동일 오버레이 + Cohere 최종 인식. 한영 혼용·무음 실패를 별도 집계.
- **D:** B 또는 C의 성능이 부족할 때 Fun-ASR-MLT-Nano/turbo 등 추가 후보를 비교.

한 번에 최종 STT 후보 하나만 적재하고, 모델·리비전·런타임·정밀도·스레드 수·청크·언어·사전 크기·워밍업 여부를 기록한다. 1/3/10/30초와 상한 근처 발화를 batch 1로 측정한다. 첫 Partial 지연, 종료 flush, 2차 인식, Fast, LLM 프리필/생성, 실제 앱 입력 지연을 분리한다.

## 10. 사용자의 네 가지 질문에 대한 독립 결론

1. **퍼포먼스:** Qwen 컨텍스트를 활용하는 최종 인식과 LLM 없는 Verbatim을 우선 비교하겠다. Nemotron은 유망하지만 정식판 hotword 지원·155H 지연을 과대평가하면 안 된다. 무료 모델의 최종 순위는 개인 발화·개발용어·batch 1 실측으로 결정한다.
2. **불필요한 내용:** 원 기획 이력, 2단계 세부 설계, 처음부터 7개 보정 클래스, 전 후보의 제품 어댑터 구현, 모든 발화의 무조건 재인식·LLM 처리는 줄일 수 있다. 평가·원문 보존·최소 안전장치는 줄이면 안 된다.
3. **설계·단계 정확성:** 큰 구조는 타당하지만 그대로 확정하기에는 부족하다. EMBED로 빠진 단계 표, Qwen/Cohere 의존성 충돌, 최종 경로 기준 게이트, 30초 버퍼와 긴 발화, 전체 deadline, 평가셋 분리를 먼저 고친다.
4. **회사 PC 범위:** 1-0~1-3의 기능 개발·CPU 평가와 1-4의 기능/정확도, 1-5 대부분을 수행할 수 있다. RTX 5080 동시 적재·GPU 지연은 집에서 검증한다. 회사 PC 자체의 경량 Verbatim 실사용 가능성도 열려 있으나 지연 달성 여부는 **미측정**이다.

최종 판정: **Claude 검토서는 부분 동의. 구조 단순화와 평가 우선 방향은 유용하지만, Qwen 공개 API·Nemotron 릴리스·처리량의 지연 환산 오류를 수정한 뒤 사용해야 한다.**
