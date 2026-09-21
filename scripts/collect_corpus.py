"""
한국어 위키백과에서 GraphRAG용 코퍼스를 모은다.
시드 문서에서 2홉까지 넓혀 data/docs/ 에 md 파일로 저장하고,
data/manifest.json 에 채택/탈락 기준과 건수를 남긴다.

실행: python scripts/collect_corpus.py
"""
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

API = "https://ko.wikipedia.org/w/api.php"
USER_AGENT = "GraphRAG-CorpusBuilder/1.0 (educational project; contact via GitHub issue)"
SLEEP_SEC = 0.8
MAX_RETRIES = 5
TARGET_DOC_COUNT = 60
MIN_BODY_CHARS = 800
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.json"

# AI 기업 · 모델 · 제품 생태계 — 회사 / 인물 / 모델 / 제품을 고루 섞은 코어 엔티티
SEEDS = [
    # 회사
    "OpenAI", "딥마인드", "앤트로픽", "마이크로소프트", "메타 (기업)",
    "엔비디아", "구글", "테슬라 (기업)", "xAI (기업)", "허깅 페이스",
    "Stability AI", "미스트랄 AI", "Cohere",
    # 인물
    "일론 머스크", "샘 올트먼", "순다르 피차이", "사티아 나델라",
    "마크 저커버그", "데미스 허사비스", "젠슨 황",
    # 모델
    "GPT-4", "챗GPT", "제미니 (언어 모델)", "구글 바드",
    # 제품
    "깃허브 코파일럿", "미드저니", "스테이블 디퓨전", "퍼플렉시티 (검색 엔진)",
]

# 후보에서 걸러낼 제목 패턴 (연도 · 목록 · 동음이의)
EXCLUDE_TITLE_PATTERNS = [
    re.compile(r"^\d{1,4}년(대)?$"),
    re.compile(r"^\d{1,2}월 \d{1,2}일$"),
    r"목록",
    r"동음이의",
    r"^분류:",
    r"^틀:",
]

# 분류 겹침 비교에서 제외할 "위키백과 관리·유지보수용" 분류 키워드.
# 이런 분류는 주제와 무관하게 거의 모든 문서에 붙어서 "개체가 겹친다"는 신호를 오염시킨다.
MAINTENANCE_CATEGORY_KEYWORDS = [
    "위키백과", "위키데이터", "문서", "링크", "인용 오류", "CS1",
    "출처", "정확성", "중립성", "고아", "각주",
]


def _is_substantive_category(cat: str) -> bool:
    return not any(kw in cat for kw in MAINTENANCE_CATEGORY_KEYWORDS)


def _exclude_by_title(title: str) -> bool:
    for pat in EXCLUDE_TITLE_PATTERNS:
        if isinstance(pat, re.Pattern):
            if pat.match(title):
                return True
        elif pat in title:
            return True
    return False


def api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            time.sleep(SLEEP_SEC)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < MAX_RETRIES:
                retry_after = e.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                print(f"    [재시도 {attempt}/{MAX_RETRIES}] HTTP {e.code} - {wait:.1f}초 대기")
                time.sleep(wait)
                backoff *= 2
                continue
            raise


def fetch_links_and_categories(title: str):
    """시드 문서 하나의 본문 링크(ns=0)와 분류 목록을 continue 처리하며 모두 받는다."""
    links, categories = set(), set()
    params = {
        "action": "query",
        "titles": title,
        "prop": "links|categories",
        "plnamespace": 0,
        "pllimit": "max",
        "cllimit": "max",
        "redirects": 1,
    }
    while True:
        data = api_get(params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" in page:
                print(f"  [경고] 시드 문서 없음: {title}")
                continue
            for link in page.get("links", []):
                links.add(link["title"])
            for cat in page.get("categories", []):
                categories.add(cat["title"])
        if "continue" in data:
            params.update(data["continue"])
        else:
            break
    return links, categories


def fetch_categories_batch(titles: list):
    """후보 문서들의 분류를 20건씩 묶어 받는다."""
    result = {t: set() for t in titles}
    for i in range(0, len(titles), 20):
        chunk = titles[i : i + 20]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "categories",
            "cllimit": "max",
        }
        while True:
            data = api_get(params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                t = page.get("title")
                if t is None or "missing" in page:
                    continue
                for cat in page.get("categories", []):
                    result.setdefault(t, set()).add(cat["title"])
            if "continue" in data:
                params.update(data["continue"])
            else:
                break
    return result


def fetch_extract(title: str) -> str:
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
    }
    data = api_get(params)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return ""
        return page.get("extract", "") or ""
    return ""


def save_doc(title: str, categories: set, body: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = title.replace(" ", "_") + ".md"
    content = f"# {title}\n분류: {', '.join(sorted(categories))}\n\n{body}\n"
    (DATA_DIR / filename).write_text(content, encoding="utf-8")


def main():
    print(f"시드 수: {len(SEEDS)}")

    seed_categories_all = set()
    link_counter = Counter()
    missing_seeds = []

    for i, seed in enumerate(SEEDS, 1):
        print(f"[{i}/{len(SEEDS)}] 시드 조회: {seed}")
        links, categories = fetch_links_and_categories(seed)
        if not links and not categories:
            missing_seeds.append(seed)
            continue
        seed_categories_all |= {c for c in categories if _is_substantive_category(c)}
        for link in links:
            if link not in SEEDS:
                link_counter[link] += 1

    print(f"\n2홉 후보(중복 제거): {len(link_counter)}건")
    print(f"  - 시드 2개 이상이 함께 가리키는 후보: {sum(1 for c in link_counter.values() if c >= 2)}건")

    # 제목 필터 (연도/목록/동음이의 등) 먼저 적용해 카테고리 조회 비용을 줄인다
    candidates_sorted = [t for t, _ in link_counter.most_common()]
    title_excluded = [t for t in candidates_sorted if _exclude_by_title(t)]
    candidates_after_title_filter = [t for t in candidates_sorted if not _exclude_by_title(t)]
    print(f"제목 필터로 제외(연도/목록/동음이의 등): {len(title_excluded)}건")

    print(f"분류 일괄 조회 대상: {len(candidates_after_title_filter)}건")
    candidate_categories = fetch_categories_batch(candidates_after_title_filter)

    category_matched = []
    category_rejected = []
    for t in candidates_after_title_filter:
        cats = {c for c in candidate_categories.get(t, set()) if _is_substantive_category(c)}
        if cats & seed_categories_all:
            category_matched.append(t)
        else:
            category_rejected.append(t)
    print(f"시드와 분류 공유 → 채택 후보: {len(category_matched)}건")
    print(f"시드와 분류 미공유 → 탈락: {len(category_rejected)}건")

    # 다중 시드가 가리키는 후보를 우선순위로 재정렬 (이미 most_common 순서라 유지됨)
    ordered_candidates = [t for t in candidate_categories.keys() if t in set(category_matched)]
    ordered_candidates.sort(key=lambda t: link_counter[t], reverse=True)

    saved = []
    stub_rejected = []
    body_cache = {}

    # 1) 시드 문서 자체를 먼저 저장 시도
    all_pull_order = [s for s in SEEDS if s not in missing_seeds] + ordered_candidates

    for title in all_pull_order:
        if len(saved) >= TARGET_DOC_COUNT:
            break
        if title in body_cache:
            continue
        body = fetch_extract(title)
        body_cache[title] = body
        if len(body) < MIN_BODY_CHARS:
            stub_rejected.append(title)
            print(f"  [탈락-토막글 {len(body)}자] {title}")
            continue
        cats = candidate_categories.get(title, set())
        if not cats and title in SEEDS:
            # 시드 자체 분류는 위에서 못 모았으므로 별도 조회
            _, cats = fetch_links_and_categories(title)
        save_doc(title, cats, body)
        saved.append(title)
        print(f"  [저장 {len(saved)}/{TARGET_DOC_COUNT}] {title} ({len(body)}자)")

    manifest = {
        "topic": "AI 기업 · 모델 · 제품 생태계",
        "seeds": SEEDS,
        "missing_seeds": missing_seeds,
        "target_doc_count": TARGET_DOC_COUNT,
        "min_body_chars": MIN_BODY_CHARS,
        "saved_doc_count": len(saved),
        "saved_titles": saved,
        "rejection_summary": {
            "title_pattern_excluded (연도/목록/동음이의)": len(title_excluded),
            "category_mismatch_excluded": len(category_rejected),
            "stub_excluded (본문 800자 미만)": len(stub_rejected),
        },
        "rejected_titles": {
            "title_pattern": title_excluded,
            "category_mismatch": category_rejected,
            "stub": stub_rejected,
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n최종 저장: {len(saved)}건 → {DATA_DIR}")
    print(f"manifest.json 기록 완료 → {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
