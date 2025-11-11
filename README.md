# Reflection_in_NER

다중에이전트 성찰 패턴의 반복이 개체명 인식(NER) 성능에 미치는 영향을 탐구하는 실험 코드입니다.  
Analyzing the impact of Multi-agent Reflection Iteration in Named Entity Recognition.

2025년도 한글 및 한국어 정보처리 & 한국코퍼스언어학회 공동 학술대회 (2025 HCLT & KACL) 논문 코드.

[참고 링크](https://sites.google.com/view/hclt-2025/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8)

## 환경 설정
1. Python 3.10+ 권장, 가상환경 생성
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 필수 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```
3. 모델 API 키
   - `OPENROUTER_API_KEY`: OpenRouter/OpenAI 호환 모델 공통 키
   - `FRIENDLI_TOKEN`: `openrouter_exaone4_32b`를 사용할 시, Friendli 키

## 실행 파라미터
`python main_runner.py [options]`  
주요 옵션은 아래와 같습니다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--rounds` | `10` | 한 샘플을 반복 반영(reflection)할 횟수 |
| `--with_type` | `False` | 타입 정보를 포함한 평가를 수행 |
| `--max_per_split` | `200` | 각 데이터 split에서 최대 평가 샘플 수 |
| `--seed` | `42` | 샘플 추출 시드 |
| `--datasets` | `conll,wnut,ebmnlp,naver` | 사용할 데이터셋 목록(콤마 구분) |
| `--splits` | `test` | 평가 split 목록(`train`,`validation`,`test`) |
| `--model` | `` | 실행할 모델 이름(콤마 구분). `--list-models`로 확인 가능 |
| `--list-models` |  | 사용 가능한 모델 목록만 출력 후 종료 |
| `--select` | `best` | round별 결과 중 `best` 또는 `last` 선택 |
| `--outdir` | `runs_ffinal` | 결과 저장 위치 |
| `--tag` | `` | 출력 폴더·파일에 붙일 사용자 태그 |
| `--max_workers` | `8` | ThreadPoolExecutor 병렬 처리 개수 |

모델 설정은 `main_runner.py` → `get_openrouter_model_cfgs()`에 정의되어 있으며 기본적으로 다음 이름을 사용합니다.
`openrouter_gpt4omini`, `openrouter_llama4scout`, `openrouter_gptoss20b`, `openrouter_llama3_8b`, `openrouter_nemotron_9b`, `openrouter_exaone4_32b`.

## 코드 실행 예시

1. 가용 모델 목록 확인
   ```bash
   python main_runner.py --list-models
   ```
2. 단일 모델·단일 데이터셋 스팟 체크
   ```bash
   python main_runner.py \
     --model openrouter_gpt4omini \
     --rounds 1 \
     --datasets conll \
     --splits test \
     --max_per_split 1 \
     --seed 42 \
     --with_type
   ```
3. 다중 모델 일괄 실행
   ```bash
   python main_runner.py \
     --model openrouter_gpt4omini,openrouter_llama4scout,openrouter_gptoss20b,openrouter_llama3_8b,openrouter_nemotron_9b
   ```
4. 긴 실험을 백그라운드에서 돌릴 때 (`nohup` + 로그 파일)
   ```bash
   nohup python main_runner.py \
     --model openrouter_gpt4omini \
     --with_type \
     --rounds 10 \
     --seed 42 \
     > run_gpt4o_s42.log 2>&1 &
   ```
5. 자원 제약이 있는 경우 워커 수 제한
   ```bash
   nohup python main_runner.py \
     --model openrouter_exaone4_32b \
     --with_type \
     --max_workers 1 \
     > run_exaone4.log 2>&1 &
   ```

## 출력 구조
- `runs_ffinal/summary.csv`: 모델 · 데이터셋 · split별 마이크로/매크로 F1 요약
- `runs_ffinal/preds/<timestamp>/<model>/…jsonl`: 각 샘플의 prediction 로그
- `runs_ffinal/preds/<timestamp>/<model>/summary__*.json`: 라운드별 상세 통계
