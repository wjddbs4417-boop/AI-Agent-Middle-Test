# AI 사서 보조 싱글 에이전트

## 2026-11-08 보완 사항

`save_user_info`의 저장 허용 검사만 보완했으며 에이전트 구성과 수업 기반 미들웨어 구조는 유지했습니다.

- 저장 거부·취소 표현이 있으면 저장 요청보다 우선하여 차단합니다. 기존 독서 선호는 그대로 유지합니다.
- 단순히 ‘저장’·‘기억’이라는 단어가 등장하는 것만으로 저장을 허용하지 않습니다. ‘저장해줘’, ‘기억해 주세요’ 등 명시적 요청 표현을 확인합니다.
- 조회 질문, 공백을 섞은 거부, 기존 선호 보존, 정상 저장·갱신, 요청 후 취소를 포함하여 모델 호출 없는 테스트 13개가 통과했습니다.
- 규칙 기반 검사이므로 모든 자연어 표현을 판별하지는 못합니다. 모호하거나 지원하지 않는 표현은 저장을 거절할 수 있습니다. 저장하려면 ‘독서 선호로 저장해줘’라고 요청하세요.

`demo_results.json`과 `scenario_validation.txt`는 **2026-10-19 기존 버전의 실제 모델 실행 기록**입니다. 보완본에서 모델 시나리오를 재실행한 결과가 아닙니다. 이번 추가 검증 결과는 `revision_validation.txt`에 기록했습니다.

202258102 최정윤 / 인지감성AI에이전트 중간고사

수업 `project_4/project_4/05_full_agent.py`를 기반으로 메일 업무를 사서 업무로 변경했습니다. 원본은 보존했습니다.

## 실행

Python 3.12와 Ollama가 필요합니다. 이 폴더에서 실행합니다.

```powershell
python -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct-2507-q4_K_M
python 01_library_agent.py
```

Ollama 앱을 실행하거나 별도 터미널에서 `ollama serve`를 실행해 두세요. API 키와 `.env`는 필요 없습니다.

- `종료` 또는 `exit`: 종료
- `/new`: 새 대화방. 같은 프로그램 내 독서 선호는 유지
- `python 01_library_agent.py --demo`: 9개 자동 실행 점검. 영상 생성 기능이 아닙니다.
- `python -m unittest discover -s . -p "test_*.py" -v`: 모델 호출 없는 경계 검사

## 수업 코드 대응

| 수업 원본 | 수정 구현 |
|---|---|
| ChatOllama / main_model / judge_model | 지정된 Qwen 모델 사용 |
| read_email_tool | search_books, check_my_loans, library_guide |
| send_email_tool | reply_to_user: 이번 턴 도구 근거로 회신 |
| Context / enrich_context | 사서 이름·역할 및 응대 원칙 주입 |
| email_policy_guardrail | library_policy_guardrail: 타인 정보·업무 외·우회 요청 차단 |
| refund_promise_guardrail | final_answer_guardrail: Judge와 도구 원문 기반 최종 회신 |
| get_user_info / save_user_info | 이용자별 독서 선호 조회·저장 |
| InMemorySaver / InMemoryStore | 대화방 기억 / 대화방 간 선호 공유 |
| 보완 사항 | after_model에서 도구 생략 1회 재시도 및 회신 도구 연결 |

## 검증 결과

`demo_results.json`은 지정 모델로 2026-10-19 실행한 최종 결과입니다. 저자 검색, 후속 질문, 미발견 안내, 본인 대출, 개인정보 차단, 업무 외 차단, 선호 저장, 다른 대화방 선호 조회, 미제공 운영시간 안내의 9개 시나리오를 확인했습니다. 성능 벤치마크나 모든 입력에 대한 안전 보장을 뜻하지 않습니다.

## 구현 범위

도서 11권은 과제의 제목·저자 데이터입니다. 대출 내역은 **가상 실습 데이터**이며 실제 기관 정보가 아닙니다. 실제 대출·예약·반납 기능, 운영시간 데이터, 외부 API와 RAG는 사용하지 않습니다.

장기 기억은 수업과 같은 InMemoryStore이므로 프로그램 종료 시 사라집니다. 사용자 ID는 데모 코드에서 지정하며 실제 인증 기능은 없습니다. 키워드 가드레일은 미등록 표현을 놓치거나 정상 질의를 차단할 수 있습니다. 최종 회신은 생성 문장을 그대로 내보내지 않고 성공한 도구 원문으로 제한합니다. 선호 저장은 최신 값으로 갱신합니다.

실행 환경의 LangChain/Pydantic 조합에서 ToolRuntime 직렬화 경고가 표시될 수 있으나, 최종 테스트에서는 도구 실행 및 기억 저장·조회가 완료되었습니다. 경고를 숨기지는 않았습니다.

## 데모 영상

[AI 사서 보조 에이전트 실행 영상](https://youtu.be/2tzZyg7n5zA?si=dL0Npg03zF4Asod6)

## 파일 구성

- `01_library_agent.py`: 에이전트 실행 코드
- `requirements.txt`: 의존 라이브러리
- `test_library_agent.py`: 모델 호출 없는 단위 테스트 13개
- `demo_results.json`, `scenario_validation.txt`: 2026-10-19 기존 버전의 모델 실행 기록
- `revision_validation.txt`: 2026-11-08 보완본의 검증 기록
- `.gitignore`: 환경 설정과 캐시 파일 제외 규칙
