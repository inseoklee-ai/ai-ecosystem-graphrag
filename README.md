# AI 기업 · 모델 · 제품 생태계 GraphRAG 챗봇

한국어 위키백과 문서 60건에서 지식 그래프를 뽑아, 질문마다 시작 개체를 찾고 n홉까지
확장한 뒤 **근거로 조회된 사실만으로** 답하고 경로·근거를 함께 보여주는 GraphRAG 에이전트입니다.
근거가 없으면 지어내지 않고 "모른다"고 답합니다.

```
문서 60건 → ① 추출 → ② 정제·병합 → 지식 그래프(노드 608 · 엣지 702)
                                          ↓
    질문 → ③ 시작 개체 찾기 → ④ n홉 확장 → ⑤ 근거만으로 답변 + 경로 제시
                                  ↓
                끊기면 넓히고(2홉→3홉), 그래도 없으면 모른다고 말하기
```

## 환경 준비

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

OpenAI API 키가 필요합니다. 이 프로젝트는 `keys.env`를 별도 경로에서 읽습니다
(레포에 키를 커밋하지 않기 위함). `build_graph.py`, `agent.py`, `evaluate.py` 상단의
`load_dotenv(...)` 경로를 자신의 키 파일 경로로 바꾸거나, 프로젝트 루트에 `.env`를 만들고
`OPENAI_API_KEY=...`를 넣은 뒤 `load_dotenv()`로 바꿔 쓰세요.

## 실행 순서 (파이프라인 전체)

```bash
# 1. 코퍼스 수집 (위키백과 API, 시드 28개 → 2홉 확장 → 60건)
python scripts/collect_corpus.py

# 2. 그래프 추출 + 정규화 (LLM 스키마 추출 → 별칭 병합 → graph.graphml)
python build_graph.py

# 3. 에이전트 단독 실행 (커맨드라인에서 질문 하나 테스트)
python agent.py "엔비디아는 누가 설립했나요?"

# 4. 평가 (골든셋 14문항 채점 + basic RAG 대조)
python evaluate.py

# 5. 웹 데모
streamlit run app.py
```

각 단계 산출물은 `output/`, `data/`에 남고, `build_graph.py`는 캐시(`output/triples_raw.json`)가
있으면 API를 다시 호출하지 않습니다.

## 데모 화면

![데모 화면 - 답변과 경로](docs/screenshot_answer.png)
![데모 화면 - 거부 사례](docs/screenshot_refuse.png)

왼쪽 사이드바의 예시 질문을 클릭하면 입력창에 채워집니다. "질문하기"를 누르면:
- **답변**: 근거로 조회된 삼중항만으로 만든 답
- **시작 개체 / 사용한 홉 수 / 근거 부족 여부**: 탐색이 어떻게 이뤄졌는지 요약
- **탄 경로**: 실제로 탄 관계 사슬 (개체 --[관계]--> 개체)
- **근거 삼중항 · 출처 문서**: 답을 뒷받침하는 원문 문장과 문서명

## 프로젝트 구조

```
ai-ecosystem-graphrag/
├── data/
│   ├── docs/            # 위키백과 원본 문서 60건
│   ├── manifest.json    # 코퍼스 채택/탈락 기준과 건수
│   └── goldenset.json   # 평가셋 14문항 (기대 정답·경로·근거)
├── config.json           # 스키마(노드·관계) · 추출 파라미터 · 정규화 별칭 · 탐색 반경/상한
├── scripts/collect_corpus.py
├── build_graph.py         # 추출 + 정제
├── agent.py                # 시작 개체 → n홉 확장 → 답변 (LangGraph, 경로 기록)
├── evaluate.py              # 홉 수별 채점 + basic RAG 대조
├── app.py                    # Streamlit 데모
└── output/
    ├── graph.graphml
    ├── triples_normalized.json
    ├── runs.jsonl        # agent.py가 답한 모든 질문 로그
    └── eval_results.json
```

## 참고: 데이터 출처

`scripts/collect_corpus.py`가 한국어 위키백과 MediaWiki API로 수집합니다. 채택/탈락 기준은
`data/manifest.json`에 남아 있습니다 (연도/목록/동음이의 문서 제외, 위키백과 관리용 분류는
개체 겹침 판단에서 제외, 본문 800자 미만 토막글 제외).
