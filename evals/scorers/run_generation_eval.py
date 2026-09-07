"""Generator eval: faithfulness and answer relevancy over oracle context.

Run as a module from the repo root:

    python -m evals.scorers.run_generation_eval --save
    python -m evals.scorers.run_generation_eval --n 8            # smoke test
    python -m evals.scorers.run_generation_eval --ids q129 q291   # specific rows
    python -m evals.scorers.run_generation_eval --concurrency 2 --throttle 1.0

No Pinecone, no embeddings. Context is rebuilt from the golden set's gold spans
(see oracle_context.py), so the only network calls are the generator and the
judge.

What the two numbers mean
-------------------------
Both come from DeepEval and both are LLM-judged, so neither is arithmetic the
way the retrieval metrics were. Read them as estimates with error bars.

  faithfulness   The judge splits the answer into atomic claims and asks, per
                 claim, whether the context supports it. Score = supported /
                 total. This is the hallucination metric. Low score = the model
                 asserted things the transcripts do not say.

  answer         The judge splits the answer into statements and asks, per
  relevancy      statement, whether it addresses the question. Score =
                 relevant / total. This is the waffle metric. Low score = the
                 model answered a nearby question, or padded.

They fail in opposite directions, which is why you need both. A model that
copies a whole context block verbatim scores ~1.0 faithful and poorly relevant.
A model that answers crisply from memory scores well on relevancy and badly on
faithfulness.

Why the citation markers are stripped
-------------------------------------
The generator emits `[c1: "verbatim quote"]` inline. Those quotes are copied
character-for-character from the context, so leaving them in inflates
faithfulness (the judge sees claims that trivially match the source) and
depresses relevancy (quote fragments read as statements that do not answer the
question). Metrics therefore run on `strip_answer(...)`. The raw answer is still
used for the citation block below, which needs the markers.

Pass --raw-answer to score the unstripped text and see the gap for yourself.

The citation block is free
--------------------------
Citations.resolve/verify_quote is arithmetic -- longest contiguous token run
against the cited chunk. It costs nothing and it measures something the judge
cannot: whether the model's *stated evidence* actually exists. Read the two
together:

  faithful high + verify high  -> healthy
  faithful high + verify low   -> right answer, invented quotes. A UI bug
                                  waiting to happen, since the app links those.
  faithful low  + verify high  -> quotes real, inferences beyond them
  refusal_rate > 0             -> OVER-refusal. Oracle context is the gold span
                                  itself, so any refusal here is the generator
                                  declining to answer a question it was handed
                                  the answer to.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from statistics import mean
from typing import Literal


def _preset_env() -> None:
    """Environment DeepEval reads at import time, so argparse is too late.

    DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE is the one that matters. Its
    default budget assumes a judge that answers immediately; ours deliberately
    waits on a token bucket, and a metric killed mid-wait reports
    "Timed out/cancelled while evaluating metric" -- which reads like a bug in
    the judge rather than the eval being paced. Hence the pre-scan of argv.
    """
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    default = os.getenv("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1800")
    try:
        default = sys.argv[sys.argv.index("--task-timeout") + 1]
    except (ValueError, IndexError):
        pass
    os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = str(default)


_preset_env()

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import (  # noqa: E402
    AsyncConfig,
    CacheConfig,
    DisplayConfig,
    ErrorConfig,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.scorers.deepeval_judge import GroqJudge  # noqa: E402
from evals.scorers.oracle_context import build_cases, stratified_sample  # noqa: E402
from src.core.config import PROJECT_ROOT, settings  # noqa: E402
from src.core.logging import get_logger  # noqa: E402
from src.graph.nodes import NO_CONTEXT_MESSAGE, Generate_node  # noqa: E402
from src.rag.Citations import build_context, resolve, strip_answer  # noqa: E402

log = get_logger(__name__)

DATASET = PROJECT_ROOT / "evals" / "datasets" / "golden_dataset_SD.json"
RESULTS = PROJECT_ROOT / "evals" / "results"

# The generator's own "I don't know" string, from Generate_prompt.
REFUSAL_MARK = "couldn't find enough information"


# ---------------------------------------------------------------
# generation
# ---------------------------------------------------------------

def _docs(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c.get("text", ""),
            metadata={k: v for k, v in c.items() if k != "text"},
        )
        for c in chunks
    ]


async def _generate_one(row: dict, sem: asyncio.Semaphore, retries: int) -> str | None:
    """One answer, through the PRODUCTION node.

    Calling Generate_node rather than re-assembling Generate_prompt | get_llm()
    is deliberate: a copy of the generator in the eval is a generator whose
    prompt can drift from the one that ships, and the eval would keep passing
    while production regressed.
    """
    async with sem:
        for attempt in range(retries):
            try:
                out = await Generate_node(
                    {"query": row["query"], "chunks": row["chunks"]}
                )
                return out["messages"][0].content
            except Exception as e:
                wait = 2 ** attempt
                log.warning(
                    f"{row['id']}: generation failed "
                    f"({type(e).__name__}: {e}); retry in {wait}s"
                )
                await asyncio.sleep(wait)
        log.error(f"{row['id']}: giving up after {retries} attempts")
        return None


async def _generate_all(rows: list[dict], concurrency: int, retries: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    answers = await asyncio.gather(
        *(_generate_one(r, sem, retries) for r in rows)
    )
    return [
        {**r, "answer": a} for r, a in zip(rows, answers) if a is not None
    ]


# ---------------------------------------------------------------
# deterministic citation stats
# ---------------------------------------------------------------

def citation_stats(row: dict) -> dict:
    """Arithmetic grounding signals. No judge, no cost.

    build_context is deterministic given chunk order, so the id map rebuilt here
    is the same one the generator saw -- exactly how cite_node does it.
    """
    _, cmap = build_context(_docs(row["chunks"]))
    resolved, hallucinated = resolve(row["answer"], cmap)

    quoted = [c for c in resolved if c.get("verified") is not None]
    verified = [c for c in quoted if c["verified"]]
    used = {(c["video_id"], c["chunk_index"]) for c in resolved}

    return {
        "n_blocks": len(cmap),
        "n_citations": len(resolved),
        "n_quoted": len(quoted),
        "n_verified": len(verified),
        "n_hallucinated_ids": len(hallucinated),
        "block_utilization": len(used) / len(cmap) if cmap else 0.0,
        "refused": REFUSAL_MARK in row["answer"] or row["answer"].strip() == NO_CONTEXT_MESSAGE,
        "answer_chars": len(row["answer"]),
    }


# ---------------------------------------------------------------
# reporting
# ---------------------------------------------------------------

def _pick(metrics_data, prefix: str):
    for m in metrics_data or []:
        if m.name.lower().startswith(prefix):
            return m
    return None


def collect(result, rows: list[dict]) -> list[dict]:
    """Join DeepEval's per-case output back onto our rows, by test case name."""
    by_id = {r["id"]: r for r in rows}
    out = []
    for tr in result.test_results:
        row = by_id.get(tr.name)
        if row is None:
            continue
        faith = _pick(tr.metrics_data, "faith")
        relev = _pick(tr.metrics_data, "answer relev")
        out.append({
            "id": row["id"],
            "query_type": row["query_type"],
            "query": row["query"],
            "n_gold": len(row.get("gold_spans", [])),
            "faithfulness": None if faith is None else faith.score,
            "faithfulness_reason": None if faith is None else faith.reason,
            "faithfulness_error": None if faith is None else faith.error,
            "relevancy": None if relev is None else relev.score,
            "relevancy_reason": None if relev is None else relev.reason,
            "relevancy_error": None if relev is None else relev.error,
            "answer": row["answer"],
            **row["stats"],
        })
    return sorted(out, key=lambda r: r["id"])


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else float("nan")


def _f2(x: float) -> str:
    """Render an unscorable cell as '-' rather than 'nan'.

    A column of nan reads like a formatting bug. It is not -- it means the judge
    never returned a score for those rows, which is a much more interesting
    failure and gets its own section below the table.
    """
    return f"{'-':>7}" if x != x else f"{x:7.2f}"


def summarize(rows: list[dict], faith_t: float, relev_t: float) -> str:
    lines = [
        f"{'':14} {'n':>4} {'scored':>7} {'faith':>7} {'relev':>7} "
        f"{'f>=' + f'{faith_t:g}':>7} {'r>=' + f'{relev_t:g}':>7}"
    ]

    def line(label, sub):
        if not sub:
            return
        fp = [r for r in sub if r["faithfulness"] is not None]
        rp = [r for r in sub if r["relevancy"] is not None]
        scored = len([r for r in sub if r["faithfulness"] is not None
                      and r["relevancy"] is not None])
        lines.append(
            f"{label:14} {len(sub):>4} {scored:>7} "
            f"{_f2(_mean(sub, 'faithfulness'))} "
            f"{_f2(_mean(sub, 'relevancy'))} "
            f"{_f2(sum(r['faithfulness'] >= faith_t for r in fp) / len(fp) if fp else float('nan'))} "
            f"{_f2(sum(r['relevancy'] >= relev_t for r in rp) / len(rp) if rp else float('nan'))}"
        )

    line("overall", rows)
    lines.append("")
    lines.append("by query type:")
    for qtype in sorted({r["query_type"] for r in rows}):
        line("  " + qtype, [r for r in rows if r["query_type"] == qtype])

    n_cit = sum(r["n_citations"] for r in rows)
    n_quo = sum(r["n_quoted"] for r in rows)
    n_ver = sum(r["n_verified"] for r in rows)

    lines += [
        "",
        "citations (deterministic, no judge):",
        f"  cite_rate          {mean(r['n_citations'] > 0 for r in rows):.2f}"
        "   answers carrying at least one citation",
        f"  quoted_cite_rate   {(n_quo / n_cit if n_cit else 0):.2f}"
        "   citations carrying a checkable quote",
        f"  quote_verify_rate  {(n_ver / n_quo if n_quo else 0):.2f}"
        "   quotes found verbatim in the cited chunk",
        f"  block_utilization  {mean(r['block_utilization'] for r in rows):.2f}"
        "   context blocks the answer actually used",
        f"  hallucinated_ids   {sum(r['n_hallucinated_ids'] for r in rows):>4}"
        "   cited ids that were never in context",
        f"  over_refusal_rate  {mean(r['refused'] for r in rows):.2f}"
        "   refusals despite being handed the gold span",
    ]
    return "\n".join(lines)


def error_report(rows: list[dict]) -> str:
    """Why rows are missing from the means.

    This section exists because the first run of this harness reported `nan`
    across the whole faithfulness column and said nothing else. ErrorConfig
    ignore_errors=True keeps one bad row from killing the other 39, but it turns
    a total judge failure into a table of blanks. An eval that fails silently is
    worse than one that crashes: the crash you fix, the blanks you explain away.
    """
    blocks = []
    for key in ("faithfulness", "relevancy"):
        unscored = [r for r in rows if r[key] is None]
        if not unscored:
            continue

        counts: dict[str, int] = {}
        for r in unscored:
            msg = str(r.get(f"{key}_error") or "").strip()
            msg = msg[:300] if msg else (
                "no error recorded -- metric skipped, usually a missing "
                "test-case field"
            )
            counts[msg] = counts.get(msg, 0) + 1

        blocks.append(f"  {key}: {len(unscored)}/{len(rows)} rows unscored")
        for msg, n in sorted(counts.items(), key=lambda kv: -kv[1])[:3]:
            blocks.append(f"      x{n}  {msg}")

    if not blocks:
        return ""
    return (
        "JUDGE ERRORS -- unscored rows are excluded from every mean above:\n"
        + "\n".join(blocks)
    )


def worst(rows: list[dict], key: str, n: int = 10) -> str:
    scored = [r for r in rows if r[key] is not None]
    if not scored:
        return f"worst {n} by {key}: nothing scored (see JUDGE ERRORS)"
    scored.sort(key=lambda r: r[key])
    lines = [f"worst {min(n, len(scored))} by {key}:"]
    for r in scored[:n]:
        lines.append(
            f"  {r['id']}  {key}={r[key]:.2f}  "
            f"[{r['query_type']}] {r['query'][:60]}"
        )
        reason = r.get(f"{key}_reason") or ""
        if reason:
            lines.append(f"        {reason[:200]}")
    return "\n".join(lines)


def preflight(rows: list[dict], tpm: int) -> str:
    """Say up front what this run will cost and how long it will take.

    A 40-row sweep against an 8000 TPM budget is a 40-minute job, and finding
    that out by watching a progress bar not move is a bad way to learn it. The
    estimate is deliberately rough -- its job is to catch the order of
    magnitude, not to be right to the token.

    Per row, roughly: faithfulness sends the context once for truths and again
    inside the verdict step, plus two short calls; relevancy sends the answer
    twice. Call it three context-sized passes.
    """
    ctx = sum(sum(len(c["text"]) for c in r["chunks"]) for r in rows)
    est = int(ctx / 3.5 * 3) + 2500 * len(rows)
    minutes = est / max(tpm, 1)
    return (
        f"est. judge cost ~{est // 1000}k tokens; at {tpm} TPM that is "
        f"~{minutes:.0f} min of wall clock"
        + ("  <-- consider --n smaller or --max-blocks lower" if minutes > 25 else "")
    )


# ---------------------------------------------------------------
# judge capability probe
# ---------------------------------------------------------------

class _ProbeVerdict(BaseModel):
    verdict: Literal["yes", "no", "idk"]
    reason: str


class _Probe(BaseModel):
    """Shaped like the schemas DeepEval actually sends: a list of objects with
    a constrained field. A judge that can return a bare {"ok": true} may still
    fail on this, so probing with a toy schema proves nothing."""
    verdicts: list[_ProbeVerdict]


def judge_check(grader_model: str | None) -> int:
    """One structured call, printed in full. Costs ~200 tokens.

    Worth running before a full sweep: a checkpoint that cannot do structured
    output produces a table of blanks 40 rows later, and this tells you in five
    seconds instead.
    """
    judge = GroqJudge(model_name=grader_model)
    print(f"judge = {judge.get_model_name()}")
    prompt = (
        'Return JSON of the form {"verdicts": [{"verdict": "yes", "reason": '
        '"..."}]} with exactly two verdicts, one "yes" and one "no".'
    )
    try:
        out = judge.generate(prompt, schema=_Probe)
    except Exception as e:
        print(f"FAILED on every mode: {type(e).__name__}: {e}")
        print("\nThe judge cannot be used. Try a different GRADER_MODEL.")
        return 1

    print(f"mode  = {judge._mode}")
    print(f"type  = {type(out).__name__}")
    print(f"value = {out}")
    print(f"stats = {judge.stats}")
    if judge._mode == "raw":
        print(
            "\nNote: 'raw' means deepeval parses free text into its schemas. "
            "That works, but expect occasional parse failures on the longer "
            "faithfulness prompts."
        )
    return 0


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40,
                    help="stratified sample size (default 40)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sample seed; keep it fixed or runs are incomparable")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="run these row ids instead of sampling")
    ap.add_argument("--grader-model", default=None,
                    help="override GRADER_MODEL for this run")
    ap.add_argument("--tpm", type=int, default=8000,
                    help="judge tokens-per-minute budget. Groq on-demand gives "
                         "gpt-oss-20b 8000; check your tier before raising it")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="parallel calls. The token bucket is the real limit, "
                         "so raising this mostly increases queueing")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="extra seconds between judge calls (the bucket paces "
                         "already; this is a blunt second lever)")
    ap.add_argument("--task-timeout", type=float, default=1800,
                    help="DeepEval per-task timeout. Must exceed the time a "
                         "call can spend waiting on the token bucket")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--faith-threshold", type=float, default=0.8)
    ap.add_argument("--relevancy-threshold", type=float, default=0.7)
    ap.add_argument("--truths-limit", type=int, default=None,
                    help="cap how many 'truths' faithfulness extracts from the "
                         "context; the cheapest way to cut judge tokens")
    ap.add_argument("--max-blocks", type=int, default=6,
                    help="context blocks per row. Every block is sent to the "
                         "judge twice, so this drives the token bill")
    ap.add_argument("--min-overlap-ms", type=int, default=1000)
    ap.add_argument("--raw-answer", action="store_true",
                    help="score the answer WITH citation markers (see docstring)")
    ap.add_argument("--verbose-metrics", action="store_true",
                    help="print the judge's claim/verdict breakdown per row")
    ap.add_argument("--judge-check", action="store_true",
                    help="make one structured judge call and exit; use this "
                         "before a full run against an unfamiliar checkpoint")
    ap.add_argument("--save", action="store_true",
                    help="write per-row results to evals/results/")
    args = ap.parse_args()

    if args.judge_check:
        return judge_check(args.grader_model)

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))

    if args.ids:
        wanted = set(args.ids)
        selected = [d for d in dataset if d["id"] in wanted]
        missing = wanted - {d["id"] for d in selected}
        if missing:
            print(f"unknown ids: {sorted(missing)}", file=sys.stderr)
    else:
        selected = stratified_sample(dataset, args.n, args.seed)

    rows, skipped = build_cases(selected, args.min_overlap_ms, args.max_blocks)
    if skipped:
        print(f"skipped {len(skipped)} row(s) with no oracle context: "
              f"{', '.join(skipped[:10])}")
    if not rows:
        print("nothing to evaluate", file=sys.stderr)
        return 1

    judge = GroqJudge(model_name=args.grader_model, tpm=args.tpm,
                      retries=args.retries)
    print(f"{len(rows)} rows | generator={settings.llm_model} | "
          f"judge={judge.get_model_name()} | oracle context")
    print(f"context blocks: min={min(len(r['chunks']) for r in rows)} "
          f"max={max(len(r['chunks']) for r in rows)}")
    print(preflight(rows, args.tpm))
    print()

    print("generating...")
    rows = asyncio.run(_generate_all(rows, args.concurrency, args.retries))
    for r in rows:
        r["stats"] = citation_stats(r)

    cases = [
        LLMTestCase(
            name=r["id"],
            input=r["query"],
            actual_output=r["answer"] if args.raw_answer else strip_answer(r["answer"]),
            retrieval_context=[c["text"] for c in r["chunks"]],
        )
        for r in rows
    ]

    metrics = [
        FaithfulnessMetric(
            threshold=args.faith_threshold, model=judge,
            verbose_mode=args.verbose_metrics,
            truths_extraction_limit=args.truths_limit,
        ),
        AnswerRelevancyMetric(
            threshold=args.relevancy_threshold, model=judge,
            verbose_mode=args.verbose_metrics,
        ),
    ]

    result = evaluate(
        test_cases=cases,
        metrics=metrics,
        # Stamped onto the run so a results file six weeks old is still
        # readable. Generator scores move when a provider swaps a checkpoint.
        hyperparameters={
            "generator": settings.llm_model,
            "generator_temperature": settings.llm_temperature,
            "judge": judge.get_model_name(),
            "context": "oracle",
            "scored_text": "raw" if args.raw_answer else "stripped",
            "max_blocks": args.max_blocks,
        },
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=args.concurrency,
            throttle_value=args.throttle,
        ),
        display_config=DisplayConfig(
            show_indicator=True, print_results=False, inspect_after_run=False
        ),
        # One row failing its judge call should not lose the other 39.
        error_config=ErrorConfig(ignore_errors=True, skip_on_missing_params=True),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )

    scored = collect(result, rows)
    print()
    print(summarize(scored, args.faith_threshold, args.relevancy_threshold))

    errors = error_report(scored)
    if errors:
        print()
        print(errors)
    print()
    print(
        f"judge calls={judge.stats['calls']} retries={judge.stats['retries']} "
        f"failures={judge.stats['failures']} mode={judge._mode} | "
        f"~{judge.bucket.spent // 1000}k tokens, "
        f"{judge.bucket.waited / 60:.1f} min spent waiting on the TPM budget"
    )
    if judge._mode == "raw":
        print("  mode=raw means structured output was refused. Parse failures "
              "('invalid JSON') come from here -- try a different GRADER_MODEL.")
    if judge.stats["retries"] > judge.stats["calls"] / 2:
        print("  more than half of calls were retried: the TPM budget is too "
              "tight. Lower --max-blocks or --n rather than raising --tpm.")

    print()
    print(worst(scored, "faithfulness"))
    print()
    print(worst(scored, "relevancy"))

    if args.save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / "generation_rows.json"
        out.write_text(
            json.dumps(
                {
                    "config": {
                        "generator": settings.llm_model,
                        "generator_temperature": settings.llm_temperature,
                        "judge": judge.get_model_name(),
                        "n": len(scored),
                        "seed": args.seed,
                        "context": "oracle",
                        "max_blocks": args.max_blocks,
                        "min_overlap_ms": args.min_overlap_ms,
                        "scored_text": "raw" if args.raw_answer else "stripped",
                    },
                    "rows": scored,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
