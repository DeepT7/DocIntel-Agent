import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from app.retrieval.retrieve import vector_search, hybrid_search
from app.core.config import DEFAULT_TOP_K
from app.core.tracing import (
    observe as _observe,
    update_current_span as _update_current_span,
    flush_traces as _flush_traces,
)

RETRIEVERS = {
    "vector": vector_search,
    "hybrid": hybrid_search,
}

CACHE_PATH = Path("output/eval_cache.json")
REPORT_PATH = Path("output/eval_report.json")
CORPUS_PATH = Path("data/chunks/corpus.jsonl")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_questions(path: str = "data/eval/questions.json") -> list[dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return doc["questions"]


def _load_section_index() -> dict[tuple[str, str], list[str]]:
    """Map (url, section_heading) -> sorted chunk_ids in the current corpus."""
    index: dict[tuple[str, str], list[str]] = {}
    if not CORPUS_PATH.exists():
        return index
    with CORPUS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            key = (chunk.get("url", ""), chunk.get("section_heading", ""))
            index.setdefault(key, []).append(chunk.get("chunk_id"))
    for ids in index.values():
        ids.sort()
    return index


def resolve_gold_by_section(questions: list[dict]) -> dict:
    """Re-point gold chunk_ids to the current corpus using (url, section_heading).

    Gold is defined by *where the answer lives*, not by a chunk_id that shifts
    whenever the chunker changes. Questions whose section is missing keep their
    stored ids and are reported.
    """
    index = _load_section_index()
    resolved = 0
    missing = []
    for question in questions:
        gold = question["gold"]
        key = (gold.get("url", ""), gold.get("section_heading", ""))
        ids = index.get(key)
        if ids:
            gold["chunk_ids"] = ids
            resolved += 1
        else:
            missing.append(question["id"])
    return {"resolved": resolved, "total": len(questions), "missing": missing}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def first_relevant_question(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    """Trả về rank (1-based) của gold đầu tiên; None nếu không có."""
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_ids:
            return rank
    return None


def score_one(retrieved_ids: list[str], gold_ids: set[str]) -> dict:
    retrieved_set = set(retrieved_ids)
    overlap = retrieved_set.intersection(gold_ids)
    rank = first_relevant_question(retrieved_ids, gold_ids)

    return {
        "hit": 1.0 if overlap else 0.0,
        "recall": len(overlap) / len(gold_ids) if gold_ids else 0.0,
        "precision": len(overlap) / len(retrieved_ids) if retrieved_ids else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
    }


# --------------------------------------------------------------------------- #
# Retrieval (có cache để không gọi lại API mỗi lần đổi metric)
# --------------------------------------------------------------------------- #
def _cache_key(retriever_name: str, top_k: int, question_id: str) -> str:
    return f"{retriever_name}|{top_k}|{question_id}"


@_observe("retriever", as_type="retriever", capture_input=False, capture_output=False)
def _run_retriever(retriever_name: str, retriever, question: str, top_k: int):
    _update_current_span(retriever=retriever_name, top_k=top_k, query=question)
    return retriever(question, top_k=top_k)


def retrieve_cached(
    retriever_name: str,
    retriever,
    question: str,
    question_id: str,
    top_k: int,
    cache: dict,
    use_cache: bool,
) -> tuple[list[str], bool]:
    """Trả về (retrieved_ids, was_cached). Retrieves top_k một lần, cache lại."""
    key = _cache_key(retriever_name, top_k, question_id)
    if use_cache and key in cache:
        return cache[key], True

    results = _run_retriever(retriever_name, retriever, question, top_k)
    retrieved_ids = [r["chunk_id"] for r in results]
    cache[key] = retrieved_ids
    return retrieved_ids, False


@_observe("evaluate_retriever", capture_input=False, capture_output=False)
def evaluate_retriever(
    questions: list[dict],
    retriever,
    retriever_name: str,
    ks: list[int],
    cache: dict,
    use_cache: bool = True,
    delay: float = 0.0,
) -> list[dict]:
    """Retrieve một lần ở max(ks) rồi cắt dần để tính metric cho từng k."""
    max_k = max(ks)
    _update_current_span(retriever=retriever_name, ks=ks, n_questions=len(questions))
    rows = []

    for item in questions:
        gold_ids = set(item["gold"]["chunk_ids"])

        started = time.perf_counter()
        retrieved_ids, was_cached = retrieve_cached(
            retriever_name,
            retriever,
            item["question"],
            item["id"],
            max_k,
            cache,
            use_cache,
        )
        elapsed = time.perf_counter() - started

        row = {
            "id": item["id"],
            "category": item["category"],
            "difficulty": item["difficulty"],
            "latency_ms": None if was_cached else elapsed * 1000,
            "metrics": {str(k): score_one(retrieved_ids[:k], gold_ids) for k in ks},
        }
        rows.append(row)

        if delay and not was_cached:
            time.sleep(delay)

    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
METRIC_KEYS = ["hit", "recall", "precision", "mrr"]


def summarize_results(rows: list[dict], k: int) -> dict:
    key = str(k)
    summary = {m: statistics.fmean(r["metrics"][key][m] for r in rows) for m in METRIC_KEYS}

    latencies = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    summary["latency_ms"] = statistics.fmean(latencies) if latencies else 0.0
    summary["n"] = len(rows)
    return summary


def breakdown(rows: list[dict], k: int, field: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[field], []).append(row)
    return {value: summarize_results(group, k) for value, group in sorted(groups.items())}


def _print_summary(name: str, k: int, summary: dict) -> None:
    print(f"Retriever: {name} | k={k} | questions={summary['n']}")
    print(f"  hit@{k}      : {summary['hit']:.4f}")
    print(f"  recall@{k}   : {summary['recall']:.4f}")
    print(f"  precision@{k}: {summary['precision']:.4f}")
    print(f"  MRR@{k}      : {summary['mrr']:.4f}")
    print(f"  latency/query: {summary['latency_ms']:.2f} ms")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate retrievers on the eval question set.")
    parser.add_argument("--retriever", choices=[*RETRIEVERS, "all"], default="all")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10], help="Một hoặc nhiều giá trị k")
    parser.add_argument("--questions", type=str, default="data/eval/questions.json")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu để thử nhanh")
    parser.add_argument("--no-cache", action="store_true", help="Bỏ qua cache, gọi lại retrieval")
    parser.add_argument("--delay", type=float, default=0.0, help="Nghỉ giây giữa các câu (tránh rate limit)")
    parser.add_argument("--report", type=str, default=None, help="Đường dẫn lưu report JSON")
    parser.add_argument("--no-resolve-gold", action="store_true", help="Dùng chunk_ids cứng trong questions.json")
    args = parser.parse_args()

    ks = sorted(set(args.k))
    questions = load_questions(args.questions)
    gold_info = {"resolved": 0, "total": len(questions), "missing": [], "enabled": False}
    if not args.no_resolve_gold:
        gold_info = resolve_gold_by_section(questions)
        gold_info["enabled"] = True
        print(f"Resolved gold by section: {gold_info['resolved']}/{gold_info['total']}"
              + (f" | missing: {gold_info['missing']}" if gold_info["missing"] else ""))
    if args.limit is not None:
        questions = questions[: args.limit]

    if args.report:
        report_path = Path(args.report)
    elif args.limit is not None:
        # Tránh ghi đè báo cáo full khi chỉ chạy thử một phần
        report_path = REPORT_PATH.with_name(f"{REPORT_PATH.stem}_limit{args.limit}{REPORT_PATH.suffix}")
    else:
        report_path = REPORT_PATH

    cache = _load_cache()
    use_cache = not args.no_cache

    names = list(RETRIEVERS) if args.retriever == "all" else [args.retriever]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": {"ks": ks, "retrievers": names, "questions": len(questions), "use_cache": use_cache},
        "gold": gold_info,
        "results": {},
    }

    for name in names:
        rows = evaluate_retriever(questions, RETRIEVERS[name], name, ks, cache, use_cache, delay=args.delay)

        print("\n" + "=" * 80)
        print(f"=== {name} (n={len(rows)}) ===")
        summaries = {}
        for k in ks:
            summary = summarize_results(rows, k)
            _print_summary(name, k, summary)
            summaries[str(k)] = summary

        primary_k = ks[-1]
        print(f"\n  -- breakdown by category @k={primary_k} --")
        by_category = breakdown(rows, primary_k, "category")
        for category, summary in by_category.items():
            print(f"    {category:14s} n={summary['n']:2d}  hit={summary['hit']:.3f}  recall={summary['recall']:.3f}  mrr={summary['mrr']:.3f}")

        print(f"\n  -- breakdown by difficulty @k={primary_k} --")
        by_difficulty = breakdown(rows, primary_k, "difficulty")
        for difficulty, summary in by_difficulty.items():
            print(f"    {difficulty:14s} n={summary['n']:2d}  hit={summary['hit']:.3f}  recall={summary['recall']:.3f}  mrr={summary['mrr']:.3f}")

        report["results"][name] = {
            "summary": summaries,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "per_question": rows,
        }

    _save_cache(cache)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _flush_traces()
    print(f"\nSaved report -> {report_path}")
    print(f"Saved cache  -> {CACHE_PATH}")


if __name__ == "__main__":
    main()