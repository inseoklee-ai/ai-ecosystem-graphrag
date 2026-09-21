"""
질문 -> 시작 개체 찾기 -> n홉 확장 -> 근거만으로 답변 + 경로 제시
끊기면(근거 부족) 넓히고, max_hops까지 넓혀도 없으면 "모른다"고 답한다.

LangGraph StateGraph로 구현:
  link_start_entity -> expand_hops -> answer_from_subgraph -> (분기) -> finalize
                                            ^                     |
                                            +--- 넓히기(hop+1) ----+

실행: python agent.py "질문"  (또는 대화형으로 실행)
"""
import difflib
import json
import re
from pathlib import Path
from typing import Optional, TypedDict

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(r"C:\Users\lis29\projects\keys.env")  # 로컬 개발용. 없으면 조용히 넘어간다.

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
NORMALIZED_TRIPLES_PATH = ROOT / "output" / "triples_normalized.json"
RUNS_LOG_PATH = ROOT / "output" / "runs.jsonl"

MODEL_NAME = CONFIG["extraction"]["model"]
ALIAS_MAP = CONFIG["normalization"]["alias_map"]
DEFAULT_HOPS = CONFIG["traversal"]["default_hops"]
MAX_HOPS = CONFIG["traversal"]["max_hops"]
HUB_DEGREE_THRESHOLD = CONFIG["traversal"]["hub_degree_threshold"]
HUB_FANOUT_CAP = CONFIG["traversal"]["hub_fanout_cap"]

# 환경변수에 키가 있으면(로컬 개발) 바로 쓰고, 없으면(데모 배포) None으로 두고
# 데모 화면에서 사용자가 입력한 키로 set_api_key()를 호출해 채운다.
try:
    client: Optional[OpenAI] = OpenAI()
except Exception:
    client = None


def set_api_key(api_key: str) -> None:
    """데모(app.py)에서 사용자가 입력한 키로 클라이언트를 새로 만든다.
    이 프로세스 메모리에서만 쓰이고 디스크에 저장되지 않는다."""
    global client
    client = OpenAI(api_key=api_key)


def _require_client() -> OpenAI:
    if client is None:
        raise RuntimeError(
            "OpenAI API 키가 설정되지 않았습니다. "
            "환경변수 OPENAI_API_KEY를 설정하거나 agent.set_api_key(키)를 먼저 호출하세요."
        )
    return client


def load_graph() -> nx.MultiDiGraph:
    """graph.graphml에는 대표 근거(문서명)만 남아있어, 인용용 원문 근거 문장이 온전한
    triples_normalized.json에서 그래프를 다시 조립한다."""
    triples = json.loads(NORMALIZED_TRIPLES_PATH.read_text(encoding="utf-8"))
    g = nx.MultiDiGraph()
    for t in triples:
        g.add_node(t["subject"], type=t["subject_type"])
        g.add_node(t["object"], type=t["object_type"])
        g.add_edge(t["subject"], t["object"], relation=t["relation"], evidence=t["evidence"])
    return g


GRAPH = load_graph()
NODE_NAMES = list(GRAPH.nodes())


# ---------- 상태 정의 ----------

class Triple(TypedDict):
    subject: str
    relation: str
    object: str
    evidence: list[dict]


class AgentState(TypedDict, total=False):
    question: str
    start_entities: list[str]
    hop_used: int
    subgraph_triples: list[Triple]
    answerable: Optional[bool]
    answer: Optional[str]
    path_used: list[dict]
    evidence_used: list[dict]
    refused: bool
    refusal_reason: Optional[str]


# ---------- 엔티티 링킹 ----------

class MentionExtraction(BaseModel):
    mentions: list[str]


def normalize_name(name: str) -> str:
    name = name.strip()
    return ALIAS_MAP.get(name, name)


def extract_mentions(question: str) -> list[str]:
    completion = _require_client().beta.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "질문에서 지식 그래프 탐색을 시작할 고유명사(회사·인물·모델·제품명)만 뽑아라. "
                "조사(은/는/이/가/을/를/의/에)는 제외하고 개체명만 남겨라. "
                "'깃허브 코파일럿', '마이크로소프트 애저'처럼 회사명과 제품명이 붙어 하나의 "
                "고유한 제품/서비스 이름을 이루면 쪼개지 말고 통째로 하나의 개체로 뽑아라. "
                "최대 2개."
            )},
            {"role": "user", "content": question},
        ],
        response_format=MentionExtraction,
    )
    result = completion.choices[0].message.parsed
    return result.mentions if result else []


_PAREN_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")
NODE_BY_STRIPPED = {_PAREN_SUFFIX.sub("", n).strip(): n for n in NODE_NAMES}


def link_to_graph_node(mention: str) -> Optional[str]:
    normalized = normalize_name(mention)
    if normalized in GRAPH.nodes:
        return normalized
    # 괄호 설명이 붙은 노드는 그 설명을 뗀 이름으로 매칭한다 (예: "테슬라" -> "테슬라 (기업)").
    if normalized in NODE_BY_STRIPPED:
        return NODE_BY_STRIPPED[normalized]
    # 그 외 부분 일치는 mention이 충분히 구체적일 때만 허용한다 (짧은 일반 단어가
    # "GPU" -> "A800 GPU"처럼 엉뚱한 구체적 제품에 잘못 걸리는 사고를 막기 위함).
    if len(normalized) >= 4:
        substring_hits = [n for n in NODE_NAMES if normalized in n]
        if substring_hits:
            return min(substring_hits, key=len)
    # 유사도 기반 fallback
    close = difflib.get_close_matches(normalized, NODE_NAMES, n=1, cutoff=0.72)
    return close[0] if close else None


def find_nodes_in_text(text: str) -> list[str]:
    """질문 원문에 그래프 노드 이름이 그대로 등장하면 우선적으로 잡아낸다
    (LLM이 복합 고유명사를 잘못 쪼개는 경우에 대한 안전망). 긴 이름부터 확인해
    '코파일럿'보다 '깃허브 코파일럿'이 먼저 잡히게 한다."""
    hits = []
    for node in sorted(NODE_NAMES, key=len, reverse=True):
        if len(node) >= 2 and node in text:
            hits.append(node)
    return hits


def link_start_entity(state: AgentState) -> AgentState:
    resolved = find_nodes_in_text(state["question"])
    mentions = extract_mentions(state["question"])
    for m in mentions:
        node = link_to_graph_node(m)
        if node and node not in resolved:
            resolved.append(node)
    # 더 구체적인 이름이 이미 잡혔으면 그 이름에 포함되는 일반적인 이름은 뺀다
    # (예: "챗GPT"가 잡혔으면 그 안에 우연히 포함된 "GPT" 단독 노드는 잡음이므로 제외).
    resolved = [r for r in resolved if not any(r != other and r in other for other in resolved)]
    if not resolved:
        return {
            **state,
            "start_entities": [],
            "refused": True,
            "refusal_reason": f"시작 개체를 찾지 못함 (질문에서 추출한 개체명: {mentions}, 그래프에 없음)",
        }
    return {**state, "start_entities": resolved, "refused": False}


# ---------- n홉 확장 (허브는 상한을 두고 뻗어나감) ----------

def neighbor_edges(node: str, apply_hub_cap: bool) -> list[Triple]:
    edges: list[Triple] = []
    for _, obj, data in GRAPH.out_edges(node, data=True):
        edges.append({"subject": node, "relation": data["relation"], "object": obj,
                       "evidence": data.get("evidence", [])})
    for subj, _, data in GRAPH.in_edges(node, data=True):
        edges.append({"subject": subj, "relation": data["relation"], "object": node,
                       "evidence": data.get("evidence", [])})
    degree = GRAPH.degree(node)
    if apply_hub_cap and degree > HUB_DEGREE_THRESHOLD:
        # 허브는 완전히 막지 않고, 근거 문서 수가 많은 관계부터 상한(HUB_FANOUT_CAP)까지만 따라간다.
        # 단, 질문의 시작 개체 자신은 대상에서 제외한다 (질문이 바로 그 개체에 관한 것이므로
        # 상한을 걸면 정작 필요한 관계가 잘릴 수 있다).
        edges.sort(key=lambda e: -len(e["evidence"]))
        edges = edges[:HUB_FANOUT_CAP]
    return edges


def bfs_subgraph(start_nodes: list[str], hops: int) -> list[Triple]:
    start_set = set(start_nodes)
    visited = set(start_nodes)
    frontier = set(start_nodes)
    collected: dict[tuple, Triple] = {}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            if node not in GRAPH.nodes:
                continue
            for edge in neighbor_edges(node, apply_hub_cap=node not in start_set):
                key = (edge["subject"], edge["relation"], edge["object"])
                collected[key] = edge
                other = edge["object"] if edge["subject"] == node else edge["subject"]
                if other not in visited:
                    next_frontier.add(other)
        visited |= next_frontier
        if not next_frontier:
            break
        frontier = next_frontier
    return list(collected.values())


def expand_hops(state: AgentState) -> AgentState:
    if state.get("refused"):
        return state
    hop_used = state.get("hop_used") or DEFAULT_HOPS
    triples = bfs_subgraph(state["start_entities"], hop_used)
    return {**state, "hop_used": hop_used, "subgraph_triples": triples}


# ---------- 근거 기반 답변 ----------

class PathEdge(BaseModel):
    subject: str
    relation: str
    object: str


class AnswerResult(BaseModel):
    reasoning: str
    answerable: bool
    answer: str
    path_used: list[PathEdge]


def format_subgraph(triples: list[Triple]) -> str:
    lines = []
    for t in triples:
        ev = t["evidence"][0] if t["evidence"] else {"doc": "?", "quote": ""}
        lines.append(
            f"- {t['subject']} --[{t['relation']}]--> {t['object']}  "
            f"(출처: {ev.get('doc', '?')} / \"{ev.get('quote', '')[:80]}\")"
        )
    return "\n".join(lines)


def answer_from_subgraph(state: AgentState) -> AgentState:
    if state.get("refused"):
        return state
    context = format_subgraph(state["subgraph_triples"])
    completion = _require_client().beta.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "아래 삼중항 목록(지식 그래프에서 조회한 사실)만 근거로 질문에 답하라. "
                "목록에 없는 지식이나 추측을 사용하지 마라. "
                "같은 회사에 LEADS 관계로 이어진 사람이 여럿이면(예: 임시 대표와 정식 대표), "
                "출처 문장에서 '현재', '해임', '임시', '복귀' 같은 표현을 근거로 지금 시점에 "
                "맞는 사람을 판단해서 답하라. "
                "질문이 요구하는 모든 조건을 실제로 잇는 경로가 목록에 있을 때만 답하라. "
                "그 경로에 쓰이지 않는, 그저 근처에 있을 뿐인 개체를 답으로 쓰지 마라. "
                "단, 목록의 삼중항 2~3개를 순서대로 이어서 조건을 모두 만족시킬 수 있다면 "
                "그 체인은 정상적인 유효한 근거이니 주저 말고 사용하라. "
                "답을 뒷받침하는 삼중항들을 path_used에 순서대로 적어라. "
                "목록만으로 질문의 모든 조건을 만족하는 답을 만들 수 없으면 "
                "answerable=false로 하고 answer는 빈 문자열로 두라. "
                "reasoning에는 목록을 훑으며 질문이 요구하는 각 조건을 어떤 삼중항으로 "
                "채웠는지 단계별로 적어라 (조건이 여러 개면 조건마다 하나씩). "
                "'~사람 중 한 명'처럼 후보가 여럿일 수 있는 조건이면, 후보마다 체인을 "
                "끝까지 따라가 보고 그중 질문의 나머지 조건까지 만족하는 후보를 답으로 써라. "
                "첫 번째로 시도한 후보가 막힌다고 바로 포기하지 말고 다른 후보도 확인하라."
            )},
            {"role": "user", "content": f"[질문]\n{state['question']}\n\n[조회된 삼중항]\n{context}"},
        ],
        response_format=AnswerResult,
    )
    result = completion.choices[0].message.parsed
    if not result or not result.answerable or not result.path_used:
        return {**state, "answerable": False}

    # LLM이 고른 경로(subject, relation, object)를 실제 조회된 삼중항과 대조해
    # 원문 근거 문장을 그대로 붙인다 (LLM이 근거를 지어내거나 잘못 옮기는 것을 방지).
    lookup = {(t["subject"], t["relation"], t["object"]): t["evidence"] for t in state["subgraph_triples"]}
    path_used, evidence_used = [], []
    for p in result.path_used:
        key = (p.subject, p.relation, p.object)
        if key not in lookup:
            continue
        path_used.append(p.model_dump())
        evidence_used.extend(lookup[key])

    if not path_used:
        return {**state, "answerable": False}

    # 경로가 실제로 하나의 사슬을 이루는지 확인한다: 시작 개체에서 출발해
    # 각 삼중항이 앞 삼중항과 개체를 공유하며 이어져야 한다. 존재하는 사실들을
    # 앞뒤로 이어지지 않게 짜깁기해 답을 지어내는 것을 막기 위함이다.
    touched = {p["subject"] for p in path_used} | {p["object"] for p in path_used}
    if not touched & set(state["start_entities"]):
        return {**state, "answerable": False}
    reached = set(state["start_entities"])
    remaining = list(path_used)
    changed = True
    while changed and remaining:
        changed = False
        for p in list(remaining):
            if p["subject"] in reached or p["object"] in reached:
                reached.add(p["subject"])
                reached.add(p["object"])
                remaining.remove(p)
                changed = True
    if remaining:
        return {**state, "answerable": False}

    return {
        **state,
        "answerable": True,
        "answer": result.answer,
        "path_used": path_used,
        "evidence_used": evidence_used,
    }


def route_after_answer(state: AgentState) -> str:
    if state.get("refused"):
        return "finalize"
    if state.get("answerable"):
        return "finalize"
    if state.get("hop_used", DEFAULT_HOPS) < MAX_HOPS:
        return "widen"
    return "give_up"


def widen(state: AgentState) -> AgentState:
    return {**state, "hop_used": state["hop_used"] + 1}


def give_up(state: AgentState) -> AgentState:
    return {
        **state,
        "refused": True,
        "refusal_reason": f"{MAX_HOPS}홉까지 넓혀도 근거를 찾지 못함",
        "answer": "모르겠습니다. 코퍼스 안에서 이 질문에 답할 근거를 찾지 못했습니다.",
    }


def finalize(state: AgentState) -> AgentState:
    if state.get("refused") and not state.get("answer"):
        state = {**state, "answer": f"모르겠습니다. ({state.get('refusal_reason', '근거 없음')})"}
    log_entry = {
        "question": state["question"],
        "start_entities": state.get("start_entities", []),
        "hop_used": state.get("hop_used"),
        "refused": state.get("refused", False),
        "refusal_reason": state.get("refusal_reason"),
        "answer": state.get("answer"),
        "path_used": state.get("path_used", []),
        "evidence_used": state.get("evidence_used", []),
    }
    RUNS_LOG_PATH.parent.mkdir(exist_ok=True)
    with open(RUNS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    return state


# ---------- LangGraph 조립 ----------

from langgraph.graph import StateGraph, END

def build_app():
    graph = StateGraph(AgentState)
    graph.add_node("link_start_entity", link_start_entity)
    graph.add_node("expand_hops", expand_hops)
    graph.add_node("answer_from_subgraph", answer_from_subgraph)
    graph.add_node("widen", widen)
    graph.add_node("give_up", give_up)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("link_start_entity")
    graph.add_conditional_edges(
        "link_start_entity",
        lambda s: "finalize" if s.get("refused") else "expand_hops",
        {"finalize": "finalize", "expand_hops": "expand_hops"},
    )
    graph.add_edge("expand_hops", "answer_from_subgraph")
    graph.add_conditional_edges(
        "answer_from_subgraph",
        route_after_answer,
        {"finalize": "finalize", "widen": "widen", "give_up": "give_up"},
    )
    graph.add_edge("widen", "expand_hops")
    graph.add_edge("give_up", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


APP = build_app()


def ask(question: str) -> AgentState:
    return APP.invoke({"question": question})


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "엔비디아는 누가 설립했나요?"
    result = ask(q)
    print(f"\n[질문] {q}")
    print(f"[시작 개체] {result.get('start_entities')}")
    print(f"[사용 홉수] {result.get('hop_used')}")
    print(f"[답변] {result.get('answer')}")
    if result.get("path_used"):
        print("[경로]")
        for p in result["path_used"]:
            print(f"  {p['subject']} --[{p['relation']}]--> {p['object']}")
    if result.get("refused"):
        print(f"[거부 사유] {result.get('refusal_reason')}")
