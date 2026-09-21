"""
골든셋(data/goldenset.json)으로 GraphRAG 에이전트를 채점하고, basic RAG(BM25)와 대조한다.

- GraphRAG: 기대 경로가 실제로 탄 경로에 얼마나 들어왔는지(경로 재현율) + 정답 일치 여부(LLM 심판)
- basic RAG: BM25로 문서 top-5를 뽑아 그 안에서만 답하게 한 뒤 같은 기준으로 채점
- 결과는 평균이 아니라 홉 수별(1/2/3/거부)로 갈라 보고한다

실행: python evaluate.py
출력: output/eval_results.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from agent import ask as graphrag_ask

load_dotenv(r"C:\Users\lis29\projects\keys.env")

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
GOLDENSET_PATH = ROOT / "data" / "goldenset.json"
DOCS_DIR = ROOT / "data" / "docs"
EVAL_OUT_PATH = ROOT / "output" / "eval_results.json"

MODEL_NAME = CONFIG["extraction"]["model"]
TOP_K = 5

client = OpenAI()


# ---------- basic RAG (BM25) ----------

def tokenize(text: str) -> list[str]:
    return re.findall(r"[가-힣]+|[A-Za-z0-9]+", text.lower())


def load_corpus():
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append({"title": path.stem, "text": path.read_text(encoding="utf-8")})
    return docs


CORPUS = load_corpus()
BM25 = BM25Okapi([tokenize(d["text"]) for d in CORPUS])


def basic_rag_answer(question: str) -> dict:
    scores = BM25.get_scores(tokenize(question))
    ranked = sorted(range(len(CORPUS)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    top_docs = [CORPUS[i] for i in ranked]
    context = "\n\n".join(f"[{d['title']}]\n{d['text'][:1500]}" for d in top_docs)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "아래 문서만 근거로 질문에 답하라. 문서에 없는 내용은 답하지 말고 "
                "정확히 '모르겠습니다.'라고만 답하라."
            )},
            {"role": "user", "content": f"[질문]\n{question}\n\n[문서]\n{context}"},
        ],
    )
    return {
        "answer": completion.choices[0].message.content.strip(),
        "retrieved_docs": [d["title"] for d in top_docs],
    }


# ---------- LLM 심판 ----------

class Judgment(BaseModel):
    correct: bool
    reason: str


def judge_answer(question: str, expected_answer: str, actual_answer: str) -> Judgment:
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "질문, 기대 정답, 실제 답변이 주어진다. 실제 답변이 기대 정답의 핵심 사실과 "
                "일치하면(표현이 달라도 괜찮다) correct=true, 다르거나 '모르겠습니다' 류로 "
                "회피했으면 correct=false로 판단하라."
            )},
            {"role": "user", "content": (
                f"[질문] {question}\n[기대 정답] {expected_answer}\n[실제 답변] {actual_answer}"
            )},
        ],
        response_format=Judgment,
    )
    return completion.choices[0].message.parsed


def is_refusal(answer: str) -> bool:
    return "모르" in answer or "찾지 못" in answer


def path_recall(expected_path: list[dict], path_used: list[dict]) -> float:
    if not expected_path:
        return None
    used_set = {(p["subject"], p["relation"], p["object"]) for p in path_used}
    hits = sum(1 for p in expected_path if (p["subject"], p["relation"], p["object"]) in used_set)
    return hits / len(expected_path)


# ---------- 메인 평가 루프 ----------

def hop_key(item: dict) -> str:
    return str(item["hop_count"]) if item["answerable"] else "거부(근거없음)"


def main():
    goldenset = json.loads(GOLDENSET_PATH.read_text(encoding="utf-8"))
    results = []

    for item in goldenset["items"]:
        print(f"[{item['id']}] {item['question']}")

        gr = graphrag_ask(item["question"])
        br = basic_rag_answer(item["question"])

        if item["answerable"]:
            gr_correct = judge_answer(item["question"], item["expected_answer"], gr.get("answer") or "").correct
            br_correct = judge_answer(item["question"], item["expected_answer"], br["answer"]).correct
            recall = path_recall(item["expected_path"], gr.get("path_used") or [])
        else:
            gr_correct = bool(gr.get("refused")) and is_refusal(gr.get("answer") or "")
            br_correct = is_refusal(br["answer"])
            recall = None

        results.append({
            "id": item["id"],
            "hop_count": item["hop_count"],
            "answerable": item["answerable"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "graphrag": {
                "answer": gr.get("answer"),
                "refused": gr.get("refused", False),
                "hop_used": gr.get("hop_used"),
                "path_used": gr.get("path_used", []),
                "correct": gr_correct,
                "path_recall": recall,
            },
            "basic_rag": {
                "answer": br["answer"],
                "retrieved_docs": br["retrieved_docs"],
                "correct": br_correct,
            },
        })
        print(f"  GraphRAG correct={gr_correct} (hop={gr.get('hop_used')}, recall={recall})"
              f" | basicRAG correct={br_correct}")

    # 홉 수별 집계 (평균 하나로 뭉개지 않는다)
    by_hop = defaultdict(lambda: {"n": 0, "graphrag_correct": 0, "basic_rag_correct": 0, "recall_sum": 0.0, "recall_n": 0})
    for r in results:
        k = str(r["hop_count"]) if r["answerable"] else "거부(근거없음)"
        bucket = by_hop[k]
        bucket["n"] += 1
        bucket["graphrag_correct"] += int(r["graphrag"]["correct"])
        bucket["basic_rag_correct"] += int(r["basic_rag"]["correct"])
        if r["graphrag"]["path_recall"] is not None:
            bucket["recall_sum"] += r["graphrag"]["path_recall"]
            bucket["recall_n"] += 1

    summary = {}
    for k, b in sorted(by_hop.items()):
        summary[k] = {
            "문항수": b["n"],
            "GraphRAG_정확도": round(b["graphrag_correct"] / b["n"], 2),
            "basicRAG_정확도": round(b["basic_rag_correct"] / b["n"], 2),
            "GraphRAG_경로재현율_평균": round(b["recall_sum"] / b["recall_n"], 2) if b["recall_n"] else None,
        }

    output = {"summary_by_hop": summary, "items": results}
    EVAL_OUT_PATH.parent.mkdir(exist_ok=True)
    EVAL_OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 홉 수별 요약 ===")
    for k, v in summary.items():
        print(f"  {k}홉: {v}")
    print(f"\n저장 -> {EVAL_OUT_PATH}")


if __name__ == "__main__":
    main()
