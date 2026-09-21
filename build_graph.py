"""
data/docs/*.md 에서 config.json의 스키마에 맞춰 지식 그래프를 추출·정제한다.

1. 문서를 청크로 쪼개 LLM으로 (subject, relation, object) 삼중항을 추출 (스키마 강제, 구조화 출력)
2. 표기 통일(별칭 병합) + 일반명사 제외 + 중복 병합으로 정규화
3. networkx 그래프로 조립해 output/ 에 저장

캐시: output/triples_raw.json 이 있으면 청크별로 이미 처리한 것은 다시 호출하지 않는다.
(config.json의 extraction.force_rebuild=true 로 전체 재추출 가능)

실행: python build_graph.py
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(r"C:\Users\lis29\projects\keys.env")

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DOCS_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "output"
RAW_CACHE_PATH = OUT_DIR / "triples_raw.json"
NORMALIZED_PATH = OUT_DIR / "triples_normalized.json"
GRAPHML_PATH = OUT_DIR / "graph.graphml"

NODE_TYPES = tuple(CONFIG["schema"]["node_types"])
RELATION_DEFS = CONFIG["schema"]["relation_types"]
RELATION_NAMES = tuple(r["name"] for r in RELATION_DEFS)

EX = CONFIG["extraction"]
MODEL_NAME = EX["model"]
CHUNK_CHARS = EX["chunk_chars"]
CHUNK_OVERLAP = EX["chunk_overlap_chars"]
MAX_WORKERS = EX["max_workers"]
FORCE_REBUILD = EX["force_rebuild"]

NORM = CONFIG["normalization"]
ALIAS_MAP = NORM["alias_map"]
STOPLIST = set(NORM["generic_noun_stoplist"])

client = OpenAI()


class Triple(BaseModel):
    subject: str
    subject_type: Literal["Company", "Person", "Model", "Product"]
    relation: Literal[
        "FOUNDED", "WORKED_AT", "LEADS", "DEVELOPED", "RELEASED",
        "BUILT_ON", "ACQUIRED", "INVESTED_IN", "PARTNERED_WITH",
        "PARENT_OF", "RUNS_ON",
    ]
    object: str
    object_type: Literal["Company", "Person", "Model", "Product"]
    evidence_quote: str


class ExtractionResult(BaseModel):
    triples: list[Triple]


def build_system_prompt() -> str:
    rel_lines = "\n".join(
        f"- {r['name']} ({r['domain']} -> {r['range']}): {r['description']}"
        for r in RELATION_DEFS
    )
    return f"""당신은 지식 그래프 추출기입니다. 주어진 한국어 위키백과 본문 조각에서
아래 스키마에 정확히 맞는 관계만 추출하세요.

[노드 타입]
{', '.join(NODE_TYPES)}

[관계 타입]
{rel_lines}

[규칙]
- 텍스트에 명시적으로 서술된 관계만 추출하세요. 추측하거나 상식으로 채우지 마세요.
- 위 11개 관계 타입에 속하지 않는 관계는 추출하지 마세요.
- subject/object는 고유명사(회사명·인명·모델명·제품명)만 사용하고, "회사", "모델", "인공지능" 같은
  일반명사는 사용하지 마세요.
- 같은 문장에서 여러 관계가 나오면 모두 추출하세요.
- evidence_quote에는 그 관계를 뒷받침하는 원문 문장을 그대로(요약하지 말고) 옮기세요.
- "~ 출신이다", "~에서 근무했다", "~에서 일했다" 처럼 과거 소속을 나타내는 표현도 WORKED_AT으로 추출하세요.
- 관계를 찾지 못하면 빈 리스트를 반환하세요.
"""


SYSTEM_PROMPT = build_system_prompt()


def load_docs():
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        title = lines[0].lstrip("# ").strip()
        body = "\n".join(lines[2:]) if len(lines) > 2 else ""
        docs.append({"title": title, "file": path.name, "body": body})
    return docs


def chunk_text(text: str, size: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def extract_chunk(doc_title: str, chunk_text_: str) -> list[dict]:
    user_prompt = f"[문서 제목: {doc_title}]\n\n{chunk_text_}"
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        temperature=EX["temperature"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ExtractionResult,
    )
    result = completion.choices[0].message.parsed
    return [t.model_dump() for t in result.triples] if result else []


def run_extraction(chunk_jobs: list[dict], cache: dict) -> dict:
    todo = [j for j in chunk_jobs if j["chunk_id"] not in cache]
    print(f"전체 청크: {len(chunk_jobs)}건, 캐시 재사용: {len(chunk_jobs) - len(todo)}건, 새로 추출: {len(todo)}건")

    def worker(job):
        try:
            triples = extract_chunk(job["doc_title"], job["chunk"])
            return job["chunk_id"], triples, None
        except Exception as e:
            return job["chunk_id"], None, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(worker, job) for job in todo]
        for future in as_completed(futures):
            chunk_id, triples, err = future.result()
            done += 1
            if err:
                print(f"  [실패 {done}/{len(todo)}] {chunk_id}: {err}")
                cache[chunk_id] = []
            else:
                cache[chunk_id] = triples
                if triples:
                    print(f"  [{done}/{len(todo)}] {chunk_id} -> {len(triples)}건")
    return cache


def normalize_entity(name: str) -> str:
    name = name.strip()
    return ALIAS_MAP.get(name, name)


def is_generic(name: str) -> bool:
    return name.strip() in STOPLIST


def normalize_triples(raw_by_chunk: dict, chunk_meta: dict) -> list[dict]:
    merged: dict[tuple, dict] = {}
    dropped_generic = 0
    dropped_self_loop = 0

    for chunk_id, triples in raw_by_chunk.items():
        meta = chunk_meta.get(chunk_id, {})
        doc_title = meta.get("doc_title", chunk_id)
        for t in triples:
            subj = normalize_entity(t["subject"])
            obj = normalize_entity(t["object"])
            if is_generic(subj) or is_generic(obj):
                dropped_generic += 1
                continue
            if subj == obj:
                dropped_self_loop += 1
                continue
            key = (subj, t["relation"], obj)
            if key not in merged:
                merged[key] = {
                    "subject": subj,
                    "subject_type": t["subject_type"],
                    "relation": t["relation"],
                    "object": obj,
                    "object_type": t["object_type"],
                    "evidence": [],
                }
            merged[key]["evidence"].append({
                "doc": doc_title,
                "quote": t["evidence_quote"],
            })

    print(f"정규화: 일반명사 제외 {dropped_generic}건, 자기참조 제외 {dropped_self_loop}건")
    print(f"고유 삼중항: {len(merged)}건")
    return list(merged.values())


def build_graph(triples: list[dict]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for t in triples:
        g.add_node(t["subject"], type=t["subject_type"])
        g.add_node(t["object"], type=t["object_type"])
        g.add_edge(
            t["subject"], t["object"],
            relation=t["relation"],
            evidence_count=len(t["evidence"]),
            sources=";".join(sorted({e["doc"] for e in t["evidence"]})),
        )
    return g


def main():
    OUT_DIR.mkdir(exist_ok=True)
    docs = load_docs()
    print(f"문서 {len(docs)}건 로드")

    chunk_jobs = []
    chunk_meta = {}
    for doc in docs:
        chunks = chunk_text(doc["body"], CHUNK_CHARS, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['file']}::chunk{i}"
            chunk_jobs.append({"chunk_id": chunk_id, "doc_title": doc["title"], "chunk": chunk})
            chunk_meta[chunk_id] = {"doc_title": doc["title"]}

    cache = {}
    if RAW_CACHE_PATH.exists() and not FORCE_REBUILD:
        cache = json.loads(RAW_CACHE_PATH.read_text(encoding="utf-8"))

    start = time.time()
    cache = run_extraction(chunk_jobs, cache)
    elapsed = time.time() - start
    RAW_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    total_raw = sum(len(v) for v in cache.values())
    print(f"\n추출 완료: {elapsed:.1f}초, 원본 삼중항 {total_raw}건 -> {RAW_CACHE_PATH}")

    triples = normalize_triples(cache, chunk_meta)
    NORMALIZED_PATH.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"정규화 결과 저장 -> {NORMALIZED_PATH}")

    g = build_graph(triples)
    nx.write_graphml(g, GRAPHML_PATH)
    print(f"그래프 저장 -> {GRAPHML_PATH}")
    print(f"노드 {g.number_of_nodes()}개, 엣지 {g.number_of_edges()}개")

    degrees = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:15]
    print("\n연결이 많은 노드 상위 15개 (허브 후보):")
    for node, deg in degrees:
        print(f"  {node} ({g.nodes[node].get('type', '?')}): {deg}")


if __name__ == "__main__":
    main()
