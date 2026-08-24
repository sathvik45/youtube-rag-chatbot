"""Run the golden set through the real retriever and print the recall curve.

Run as a module from the repo root, so `src.*` and `evals.*` both resolve:

    python -m evals.scorers.run_retrieval_eval
    python -m evals.scorers.run_retrieval_eval --limit 20            # smoke test
    python -m evals.scorers.run_retrieval_eval --score-threshold 0.3 # absent rows
    python -m evals.scorers.run_retrieval_eval --per-video-cap 2     # playlist fairness

Requires a populated Pinecone index. See evals/README.md for ingestion.

On absent rows
--------------
score_query() marks an "absent" query correct only when the retriever returns
NOTHING. A plain similarity search always returns k chunks, so without
--score-threshold all 16 absent rows score 0 and drag the averages down. That is
the metric working as designed: it is telling you the retriever has no way to
say "I don't know". Set a threshold to give it one, then tune it here -- too high
and it starts refusing real questions, which shows up as pointwise hit falling.
"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import mean

from src.core.config import PROJECT_ROOT
from evals.scorers import Retrieval_metrics as M

# Was: ROOT = Path(__file__).resolve().parent.parent, which is evals/ -- so the
# two sys.path lines that used to live here pointed at evals/src and
# evals/evals/scorers, neither of which exists. Nothing complained because the
# imports were resolving through the script's own directory and an externally
# set PYTHONPATH. Now both packages import normally and there is no sys.path
# manipulation to get wrong.
DATASET = PROJECT_ROOT / "evals" / "datasets" / "golden_dataset_SD.json"
RESULTS = PROJECT_ROOT / "evals" / "results"


def build_retrieve_fn(per_video_cap: int | None, score_threshold: float | None):
    """Adapt rag.retriever.retrieve to what evaluate() expects.

    retrieve() returns LangChain Documents; the scorers want plain metadata
    dicts with video_id / start_ms / end_ms / chunk_index. splitters.py already
    writes all four, so this is a straight unwrap.
    """
    from src.rag.retriever import retrieve

    def retrieve_fn(query: str, video_ids: list[str], k: int) -> list[dict]:
        docs = retrieve(
            query,
            video_ids,
            k=k,
            per_video_cap=per_video_cap,
            score_threshold=score_threshold,
        )
        return [d.metadata for d in docs]

    return retrieve_fn


def worst_rows(rows: list[dict], dataset: list[dict], k: int, n: int = 15) -> str:
    """The rows to actually go and look at, worst first."""
    by_id = {d["id"]: d for d in dataset}
    at_k = [r for r in rows if r["k"] == k and r["query_type"] != "absent"]
    at_k.sort(key=lambda r: (r["coverage"], r["mean_span_coverage"]))

    lines = [f"worst {n} rows at k={k} (excluding absent):"]
    for r in at_k[:n]:
        item = by_id[r["id"]]
        lines.append(
            f"  {r['id']}  cover={r['coverage']:.2f}  span={r['mean_span_coverage']:.2f}  "
            f"vid={r['video_recall']:.2f}  n_gold={r['n_gold']}  "
            f"[{r['query_type']}] {item['query'][:60]}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N queries (smoke test)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--per-video-cap", type=int, default=None)
    ap.add_argument("--score-threshold", type=float, default=None)
    ap.add_argument("--save", action="store_true",
                    help="write per-row results to evals/results/")
    args = ap.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"{len(dataset)} queries, k={args.ks}")
    if args.score_threshold is None:
        n_absent = sum(1 for d in dataset if not d["gold_spans"])
        if n_absent:
            print(f"note: no --score-threshold, so all {n_absent} absent rows "
                  f"will score 0 (see module docstring)")
    print()

    retrieve_fn = build_retrieve_fn(args.per_video_cap, args.score_threshold)
    rows = M.evaluate(dataset, retrieve_fn, ks=tuple(args.ks))

    print(M.summarize(rows, ks=tuple(args.ks)))
    print()
    mid = args.ks[len(args.ks) // 2]
    print(worst_rows(rows, dataset, mid))

    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / "retrieval_rows.json"
        out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
