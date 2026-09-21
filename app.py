"""
로컬 데모: 질문을 입력하면 GraphRAG 에이전트가 시작 개체 -> n홉 확장 -> 근거 기반 답변을
만드는 과정을 경로·근거 삼중항·출처 문서와 함께 보여준다.

실행: streamlit run app.py
"""
import streamlit as st

import agent
from agent import ask

st.set_page_config(page_title="AI 생태계 GraphRAG 챗봇", page_icon="🕸️", layout="wide")

st.title("🕸️ AI 기업 · 모델 · 제품 생태계 GraphRAG 챗봇")
st.caption(
    "위키백과 문서 60건에서 뽑은 지식 그래프(노드 608개 · 엣지 702개) 위에서 "
    "시작 개체 → n홉 확장 → 근거 기반 답변을 수행합니다. 근거가 없으면 지어내지 않고 모른다고 답합니다."
)

with st.sidebar:
    st.subheader("🔑 OpenAI API 키")
    api_key_input = st.text_input(
        "본인의 OpenAI API 키를 입력하세요",
        type="password",
        key="api_key",
        placeholder="sk-...",
        help=(
            "이 데모는 방문자 각자의 키로 동작합니다. 입력한 키는 저장되지 않고 "
            "이 브라우저 세션 동안만 서버 메모리에서 쓰이며, 탭을 닫으면 사라집니다. "
            "API 사용 요금은 키 소유자에게 청구됩니다."
        ),
    )
    st.caption(
        "키가 없으신가요? [platform.openai.com/api-keys](https://platform.openai.com/api-keys)에서 "
        "발급받을 수 있습니다. 남용을 막으려면 발급 후 지출 한도를 걸어두는 걸 권장합니다."
    )
    st.divider()

EXAMPLES = [
    "엔비디아는 누가 설립했나요?",
    "딥마인드를 인수한 회사는 어디인가요?",
    "앤트로픽에 투자한 기업 중, 딥마인드를 인수한 곳은 어디인가요?",
    "미스트랄 AI를 만든 사람 중 한 명이 이전에 몸담았던 회사는 어디이고, 그 회사는 누구에게 인수되었나요?",
    "삼성전자가 투자한 AI 스타트업은 어디인가요?",
    "테슬라와 텐센트 사이에 인수나 투자 관계가 있나요?",
]

if "question" not in st.session_state:
    st.session_state.question = ""

with st.sidebar:
    st.subheader("예시 질문")
    st.caption("클릭하면 입력창에 채워집니다")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.question = ex

question = st.text_input(
    "질문을 입력하세요",
    key="question",
    placeholder="예: 엔비디아는 누가 설립했나요?",
)
run = st.button("질문하기", type="primary")

if run and not api_key_input.strip():
    st.error("먼저 왼쪽 사이드바에 본인의 OpenAI API 키를 입력해주세요.")
elif run and question.strip():
    agent.set_api_key(api_key_input.strip())
    try:
        with st.spinner("시작 개체를 찾고, 그래프를 확장하는 중..."):
            result = ask(question)
    except Exception as e:
        st.error(f"API 호출 중 오류가 발생했습니다: {e}")
        st.stop()

    st.divider()
    st.subheader("답변")
    st.markdown(f"### {result.get('answer', '(답변 없음)')}")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    col_meta1.metric("시작 개체", ", ".join(result.get("start_entities", [])) or "-")
    col_meta2.metric("사용한 홉 수", result.get("hop_used", "-"))
    col_meta3.metric("근거 부족 여부", "거부됨" if result.get("refused") else "답변함")

    if result.get("refused"):
        st.warning(f"거부 사유: {result.get('refusal_reason', '알 수 없음')}")

    path_used = result.get("path_used") or []
    if path_used:
        st.subheader("🔗 탄 경로")
        path_str = "  →  ".join(
            [path_used[0]["subject"]] + [f"[{p['relation']}] {p['object']}" for p in path_used]
        )
        st.code(path_str, language=None)
        for p in path_used:
            st.markdown(f"- `{p['subject']}` **--[{p['relation']}]-->** `{p['object']}`")

    evidence_used = result.get("evidence_used") or []
    if evidence_used:
        st.subheader("📄 근거 삼중항 · 출처 문서")
        for e in evidence_used:
            with st.container(border=True):
                st.markdown(f"**출처:** `{e.get('doc', '?')}`")
                if e.get("quote"):
                    st.markdown(f"> {e['quote']}")

    with st.expander("이번 답변에서 조회된 서브그래프 전체 보기"):
        st.caption(f"조회된 삼중항 {len(result.get('subgraph_triples', []))}건")
        for t in result.get("subgraph_triples", [])[:200]:
            st.text(f"{t['subject']} --[{t['relation']}]--> {t['object']}")
elif run:
    st.info("질문을 입력해주세요.")
