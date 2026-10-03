import argparse 
import json 
import re 
import statistics 
from pathlib import Path 

from app.reasoning.rag_graph import run_rag, flush_traces
from app.reasoning.answer import _generate_openrouter
from app.core.text import clean_data
from app.core.prompts import build_judge_prompt

QUESTION_PATH = Path("data/eval/questions.json")
OUT_DIR = Path("output/answer_eval")
ANSWERS_PATH = OUT_DIR / "answers.json"
JUDGEMENTS_PATH = OUT_DIR / "judgements.json"
REPORT_PATH = OUT_DIR / "report.json"

CRITERIA = [
    "faithfulness",
    "answer_relevance",
    "citation_accuracy",
]

JUDGE_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"

def load_questions(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]

def load_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}

def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean_data(data), indent=2, ensure_ascii=False), encoding="utf-8")


# Step 2 - Generate answers
def generate_answers(questions, use_cache=True):
    cache = load_json(ANSWERS_PATH)
    for q in questions:
        qid = q["id"]

        if use_cache and qid in cache:
            continue
        result = run_rag(q["question"])
        cache[qid] = {
            "question": q["question"],
            "answer": result.get("answer", ""),
            "context": result.get("context", ""),
            "source_ids": [s.get("chunk_id") for s in result.get("sources", [])],
        }

        print(f"answered {qid}")
        # checkpoint after each question to prevent losing results in case of a crash
        save_json(ANSWERS_PATH, cache) # checkpoint after each 

    return cache

# Step 3 - Judge prompt 

def parse_scores(raw):
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    data = None
    try:
        data = json.loads(text)
    except Exception:
        data = None

    out = {}
    for c in CRITERIA:
        score, reason = None, ""
        if isinstance(data, dict):
            v = data.get(c)
            if isinstance(v, dict):
                score, reason = v.get("score"), v.get("reason", "")
            elif v is not None:
                score = v
            reason = data.get(f"{c}_reason", reason)
        if score is None:                      # regex fallback
            m = (re.search(rf'{c}"?\s*:\s*\{{?\s*"score"\s*:\s*(\d)', text)
                 or re.search(rf'{c}"?\s*:\s*(\d)', text))
            if m:
                score = m.group(1)
        if score is not None:
            out[c] = {"score": int(score), "reason": reason}

    if not all(c in out for c in CRITERIA):
        raise ValueError(f"Could not extract scores: {text[:200]}")
    return out

def _match_key(d, name):
    """Find a key case-insensitively; also tolerate '1. faithfulness' style."""
    for k in d:
        kl = str(k).lower()
        if name in kl:
            return k
    return None

def call_judge(prompt, model=JUDGE_MODEL):
    import os, requests
    from app.core.config import OPENROUTER_BASE_URL
    r = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "response_format": {"type": "json_object"},   # ép JSON hợp lệ
              "max_tokens": 2048},
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

    content = r.json()["choices"][0]["message"].get("content")
    if not content:
        raise RuntimeError("judge returned empty content")
    return content 

def normalize_scores(data):
    # Case A: {"faithfulness": {...}|int, ...}
    if isinstance(data, dict):
        found = {c: _match_key(data, c) for c in CRITERIA}
        if all(found.values()):
            out = {}
            for c in CRITERIA:
                val = data[found[c]]
                if isinstance(val, dict):
                    out[c] = {"score": int(val.get("score", 0)), "reason": val.get("reason", "")}
                else:
                    out[c] = {"score": int(val), "reason": ""}
            return out

    # Case B: nested one level: {"evaluation": {criterion...}}
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict):
                try:
                    return normalize_scores(v)
                except ValueError:
                    pass

    # Case C: list of {"criterion"/"name": ..., "score": ...}
    if isinstance(data, list):
        out = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            name = None
            for key in ("criterion", "name", "metric", "aspect"):
                if key in item:
                    name = str(item[key]).lower()
                    break
            for c in CRITERIA:
                if name and c in name:
                    out[c] = {"score": int(item.get("score", 0)), "reason": item.get("reason", "")}
        if all(c in out for c in CRITERIA):
            return out

    raise ValueError(f"Unrecognized judge schema: {str(data)[:200]}")

# Step 4 Score
def judge_all(questions, answers, use_cache=True):
    cache = load_json(JUDGEMENTS_PATH)
    for q in questions:
        qid = q["id"]
        if use_cache and qid in cache:
            continue
        a = answers.get(qid)
        if not a:
            continue

        raw = ""
        try: 
            raw = call_judge(
                build_judge_prompt(a["question"], a["context"], a["answer"])
            )
            scores = parse_scores(raw)
            cache[qid] = scores
            print(f"judged {qid}: " + ", ".join(f"{c}={scores[c]['score']}" for c in CRITERIA))
        except Exception as error:
            print(f"judge failed {qid}: {error!r}")
            print(f"RAW: {raw[:800]!r}")
            continue
        save_json(JUDGEMENTS_PATH, cache)
    return cache

def summarize(questions, judgements):
    rows = [(q, judgements[q["id"]]) for q in questions if q["id"] in judgements]
    summary = {}
    for c in CRITERIA:
        scores = [r[1][c]["score"] for r in rows if c in r[1]]
        summary[c] = {
            "mean": round(statistics.fmean(scores), 3) if scores else 0.0,
            "n": len(scores)
        }
    return rows, summary

def main():
    parser = argparse.ArgumentParser(description="Answer-level eval (LLM-as-judge).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--answers-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    args = parser.parse_args()


    questions=load_questions(QUESTION_PATH)
    if args.limit:
        questions = questions[:args.limit]
    use_cache = not args.no_cache

    answers = load_json(ANSWERS_PATH)
    if not args.judge_only:
        answers = generate_answers(questions, use_cache=use_cache)

    judgements = {}
    if not args.answers_only:
        judgements = judge_all(questions, answers, use_cache=use_cache)

    if judgements:
        rows, summary = summarize(questions, judgements)
        print("\n=== Answer-level summary ===")
        for c, s in summary.items():
            print(f" {c:20s} mean={s['mean']} n={s['n']}")
        worst = sorted(rows, key=lambda r: r[1]["faithfulness"]["score"])[:5]
        print("\n lowest faithfulness:")
        for q, j in worst:
            print(f" {q['id']}: faith={j['faithfulness']['score']} - {q['question'][:60]}")

        save_json(REPORT_PATH, {
            "judge_model": JUDGE_MODEL,
            "summary": summary, 
            "per_question": {q["id"]: j for q, j in rows}
        })
        print(f"\nSaved report -> {REPORT_PATH}")

    flush_traces()

if __name__ == "__main__":
    main()



    
