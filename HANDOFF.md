# HANDOFF — AI 기업·모델·제품 생태계 GraphRAG 에이전트

다른 세션에서 이어받을 때 읽는 문서. 프로젝트 배경, 환경, 오늘 검증된 내용, 남은 일을 요약한다.

## 무엇을 만들고 있나

모두의연구소 "지식그래프 에이전트 만들기" 과제. 지난주 한국영화 GraphRAG(별도 프로젝트
`graphrag-korean-cinema`)에 이어, 이번 주는 **AI 기업·모델·제품 생태계**를 주제로 새 코퍼스를
모아 지식 그래프를 만들고, LangGraph로 시작 개체→n홉 확장→근거 기반 답변을 하는 에이전트를
구축한 뒤, basic RAG(BM25)와 성능을 비교 평가하고 Streamlit 데모까지 배포했다.

**진행 방식**: 이 세션 안에서 5단계(자료수집→그래프구축→멀티홉검색→평가→웹데모)를 순서대로
진행하며 각 단계 산출물을 사용자에게 보여주고 승인받은 뒤 다음 단계로 넘어갔다. 문제를 발견하면
바로 코드를 고치고 재실행해서 검증하는 식으로 작업했다.

## 환경

- 프로젝트 루트: `C:\Users\lis29\projects\ai-ecosystem-graphrag` (GitHub: [inseoklee-ai/ai-ecosystem-graphrag](https://github.com/inseoklee-ai/ai-ecosystem-graphrag))
- Python 3.12, venv: `.venv/` (Windows, `.venv\Scripts\activate`)
- API 키: 로컬 개발 시 `C:\Users\lis29\projects\keys.env`를 직접 로드한다.
  ```python
  from dotenv import load_dotenv
  load_dotenv(r"C:\Users\lis29\projects\keys.env")
  ```
  이 파일이 없어도(예: 다른 사람 컴퓨터, Streamlit Cloud) 에이전트가 죽지 않도록
  `agent.py`의 `client`는 지연 초기화되어 있고, `app.py`는 화면에서 사용자가 직접
  키를 입력하는 BYOK(bring-your-own-key) 구조다.
- **라이브 데모**: https://ai-ecosystem-graphrag-5lpdgjw99e3pm2dyb9be36.streamlit.app/
  (Streamlit Community Cloud, 무료 플랜이라 오래 방치하면 콜드스타트로 깨어나는 데 시간이 걸림)

## 데이터

- 코퍼스: `data/docs/*.md` (한국어 위키백과 60건), `scripts/collect_corpus.py`가 MediaWiki API로
  시드 28개 → 2홉 링크 확장 → 분류 공유 필터 → 800자 미만 토막글 제외 과정을 거쳐 수집
- 채택/탈락 상세: `data/manifest.json`
- 평가셋: `data/goldenset.json` (14문항: 1홉 6·2홉 4·3홉 2·거부 2, 기대 정답·경로·원문 근거 포함)

## 검증된 핵심 결과

- **그래프**: 노드 608개·엣지 702개 (`output/graph.graphml`, `output/triples_normalized.json`)
  - 관계별 분포: RELEASED 218·WORKED_AT 105·DEVELOPED 86·ACQUIRED 74·FOUNDED 67·LEADS 59·
    PARTNERED_WITH 41·INVESTED_IN 24·PARENT_OF 13·BUILT_ON 11·RUNS_ON 4
  - 정규화(별칭 병합) 20여 건을 `config.json`의 `normalization.alias_map`에 직접 채워 넣음
    (앤트로픽/앤트로픽 PBC, 오픈AI/OpenAI, 제미나이/제미니, 클로드/Claude, 성만 추출된 "황"/"머스크" 등)
- **평가 결과** (`output/eval_results.json`, `evaluate.py`로 GraphRAG vs basicRAG(BM25) 대조):

  | 유형 | 문항수 | GraphRAG | basicRAG | 경로재현율 |
  |---|---|---|---|---|
  | 1홉 | 6 | 100% | 17% | 1.00 |
  | 2홉 | 4 | 75% | 25% | 0.75 |
  | 3홉 | 2 | 0% | 0% | 0.17 |
  | 거부 | 2 | 100% | 100% | - |

  3홉 실패 원인을 직접 진단함: q08·q12는 색인(추출) 단계 문제(희귀/간접 관계 추출 recall
  부족), q11은 생성(답변) 단계 문제(다중 후보 중 하나가 막히면 나머지를 안 찾고 포기).
  탐색(그래프 순회) 단계 실패는 0건. 자세한 분석과 구조적 이유·해법은 `REPORT.md` 6장 참고.
- **에이전트 코드 버그 3개를 golden set 검증 중 실제로 잡음** (agent.py):
  1. 허브 상한(차수>20 노드는 뻗어나갈 때 15개로 제한)이 질문의 시작 개체 자신에게도
     걸려서 필요한 엣지가 잘리던 문제
  2. "깃허브 코파일럿" 같은 복합 고유명사를 LLM이 "깃허브"+"코파일럿"으로 쪼개 엉뚱한
     노드에 연결하던 문제
  3. "GPU" 같은 흔한 단어가 "A800 GPU" 같은 구체적 제품에 잘못 매칭되던 문제 +
     근거 삼중항들이 실제로 사슬로 이어지는지 검증하는 로직 부재(할루시네이션 방지용으로 추가함)
- **Streamlit Cloud 배포 이슈 해결**: Streamlit Cloud가 Python 3.14를 쓰는데 고정해둔
  `pydantic-core==2.23.4`가 그 버전용 wheel이 없어 Rust 소스 빌드가 실패했었음.
  `runtime.txt`로 3.12 지정을 시도했으나 반영 안 됨 → `requirements.txt`의 pydantic 버전
  고정을 풀어(`pydantic>=2.9,<3`) 해결. 자세한 삽질 과정은 `REPORT.md` 6장에 기록.

## 캐시 파일 (재실행 시 API 비용 없이 로드됨)

```
output/triples_raw.json         — build_graph.py LLM 추출 원본 (chunk별 캐시, force_rebuild=false면 재호출 안 함)
output/triples_normalized.json  — 정규화된 최종 삼중항 (agent.py가 여기서 그래프를 다시 조립함)
output/graph.graphml            — 아카이브용 그래프 (agent.py 런타임에는 안 씀, evidence quote가 없어서)
output/runs.jsonl               — agent.py가 실제로 답한 모든 질문 로그 (경로·근거 포함, 계속 append됨)
output/eval_results.json        — evaluate.py 최종 채점 결과
```

## 프로젝트 구조

```
ai-ecosystem-graphrag/
├── data/{docs/, manifest.json, goldenset.json}
├── config.json          — 스키마(노드4종·관계11종)·추출 파라미터·정규화 별칭·탐색 반경/상한
├── scripts/
│   ├── collect_corpus.py       — 위키백과 코퍼스 수집
│   └── capture_screenshots.py  — README/REPORT용 데모 캡처 (개발자 전용, playwright 필요, requirements.txt엔 없음)
├── build_graph.py       — 추출 + 정제
├── agent.py             — 시작 개체 → n홉 확장 → 답변 (LangGraph, 경로 기록, BYOK)
├── evaluate.py           — 홉 수별 채점 + basicRAG(BM25) 대조
├── app.py                 — Streamlit 데모 (API 키 입력창 포함)
├── requirements.txt, runtime.txt(현재 효과 없음, 참고용으로만 남겨둠)
├── output/, docs/(스크린샷)
├── README.md              — 실행 방법, 라이브 데모 링크, 프로젝트 구조
└── REPORT.md               — 과제 제출용 리포트 (6장 구성, 아래 참고)
```

## REPORT.md 구성 (최종본)

1. 주제와 코퍼스 (선정 이유, 수집 기준)
2. 골든셋과 스키마 (뽑지 않기로 한 관계와 그 대가, 정규화 기준)
3. 측정 결과 (홉수별 대조표, 실패 사례 색인/탐색/생성 분류)
4. 파이프라인 구조도 (Mermaid, LangGraph State)
5. 데모 설계 (BYOK 이유 포함, 캡처 2장)
6. 프로젝트 회고
   - 가장 공들인 부분 + Streamlit 배포 디버깅 사례
   - **홉 수가 늘수록 취약해지는 구조적 이유** (추출 recall 곱 감소·후보 폭증·분기 조합)와 **해법** 6가지
   - **실전 함의: 어디에 투자해야 하는가** — 팔란티어 온톨로지 시스템과 비교해 "쿼리 시점
     컴퓨팅"이 아니라 "빌드타임 데이터 정합성·사람 검증"에 투자해야 한다는 전략적 결론
     (사용자가 명시적으로 요청해 추가한, 팀 차원에서 중요하게 여기는 내용)

## 아직 안 한 것 / 제안만 하고 실행 안 한 개선

- 관계에 시점 속성(`since`/`until`) 추가 — q12류 LEADS 모호성 문제의 근본 해법으로 제안만 함
- LLM 자유서술 대신 그래프 알고리즘으로 경로 후보를 결정적으로 나열하는 구조 변경 — 제안만 함
- 3홉 골든셋 문항을 10개 이상으로 확충해 재측정 — 현재 2문항이라 0%가 통계적으로 불안정
- 커뮤니티 요약(전역 질문 대응) — 과제 지시대로 이번엔 의도적으로 생략
- 피어리뷰 자체는 아직 안 함 — 데모 질문 3개(1홉/2홉/거부) 골라뒀고 화면공유로 진행 예정

## 재현/재실행 방법

```bash
cd C:\Users\lis29\projects\ai-ecosystem-graphrag
.venv\Scripts\activate
python scripts/collect_corpus.py   # 코퍼스 재수집 (보통 불필요, data/docs 이미 있음)
python build_graph.py              # 캐시 있으면 API 재호출 안 함
python agent.py "질문"              # 단발 테스트
python evaluate.py                 # 골든셋 전체 채점
streamlit run app.py               # 로컬 데모
```

## 작업 중 겪은 실수/교훈 (다음 세션 참고용)

- **코퍼스 필터의 숨은 버그**: 위키백과 관리용 분류("출처가 필요한 문서" 등)까지 "개체
  겹침" 신호로 인정해버려서 주제와 무관한 문서가 섞여 들어온 적 있음 → 관리용 분류를
  명시적으로 제외하는 필터를 추가해야 했음. 다음에 위키 기반 코퍼스를 만들 때 처음부터
  이 필터를 넣을 것.
- **정규화는 자동화만으로 안 잡힌다**: LLM 추출 결과의 표기 분열(앤트로픽/앤트로픽 PBC 등)은
  golden set으로 실제 답이 틀리는 걸 보고 나서야 발견됨. 그래프 노드 목록을 통째로 눈으로
  훑는 단계를 정규화 다음에 꼭 넣을 것.
- **에이전트 프롬프트 튜닝은 whack-a-mole이 되기 쉬움**: "할루시네이션 방지" 지시를 강하게
  넣으면 멀쩡한 답까지 거부하고, 약하게 넣으면 없는 근거로 답을 지어냄 — 매번 골든셋 14문항
  전체로 재검증하며 균형점을 찾아야 했음. 프롬프트 한 줄 바꿀 때마다 전체 재검증하는 습관을
  유지할 것.
- **모델의 실제 사고 과정을 구조화 출력에 노출시키면 디버깅이 쉬워진다**: `AnswerResult`에
  `reasoning` 필드를 추가하고 나서야 "왜 정답이 있는데도 답을 못 냈는지"(q11)를 코드 버그가
  아니라 모델의 다중 후보 탐색 한계로 정확히 좁힐 수 있었음.
- **Streamlit Cloud 배포 에러는 로그를 끝까지 읽어야 한다**: "Error installing requirements"라는
  같은 표면 에러가 실제로는 서로 다른 원인(처음엔 불필요한 langchain-openai 의존성 의심 →
  실제로는 Python 3.14/pydantic-core wheel 부재)이었음. `runtime.txt`로 Python 버전을
  지정해도 Streamlit Cloud가 반영 안 하는 경우가 있었음 — 버전 고정을 푸는 쪽이 더 확실한
  해법이었음.
