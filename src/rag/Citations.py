"""Citation building, resolution, and verification.

The model never emits timestamps -- it emits citation ids it can see in the
context, plus a verbatim quote. We resolve ids to timestamps ourselves and
verify the quote actually occurs in the cited chunk. That makes citations
checkable rather than trusted.
"""

import re
import unicodedata

from langchain_core.documents import Document

from src.core.logging import get_logger

log = get_logger(__name__)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
# Citation markers, with an optional verbatim quote:  [c2]  or  [c2: "..."]
#
# Two kinds of tolerance are deliberate:
#
# 1. Bracket variants. Models do not reliably emit ASCII brackets --
#    gpt-oss-120b returns full-width CJK brackets (U+3010/U+3011). An
#    ASCII-only pattern misses them entirely and every citation reads as
#    "not emitted".
# 2. The quote is captured as "everything up to the closing bracket" rather
#    than by matching quote characters. Models mix straight, curly and
#    guillemet quotes freely; the closing bracket is the one delimiter they
#    get right. Surrounding quote marks are stripped afterwards.
#
# The quote group is optional so a bare [c2] still resolves -- it just cannot
# be verified.
_CITE = re.compile(r"[\[【［](c\d+)(?:\s*[:：]\s*([^\]】］]+))?[\]】］]")

# Quote characters to peel off a captured quote, whatever the model reached for.
_QUOTE_CHARS = "\"'“”‘’«»「」"

# Tuned against real transcript data: exact and punctuation-variant quotes
# score 1.00, paraphrase 0.36, fabricated 0.22, generic filler 0.50.
VERIFY_THRESHOLD = 0.9


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


# ---------------------------------------------------------------
# Building context for the prompt
# ---------------------------------------------------------------

def to_citation(doc: Document, lead_in_ms: int = 2500) -> dict:
    """Resolve a chunk into a linkable citation.

    The link starts a couple of seconds early so the viewer doesn't land
    mid-sentence, but start_ms stays exact for merging and display.
    """
    meta = doc.metadata
    start_ms = int(meta["start_ms"])
    linked_s = max(0, start_ms - lead_in_ms) // 1000

    return {
        "video_id": meta["video_id"],
        "title": meta.get("video_title", ""),
        "chunk_index": meta["chunk_index"],
        "start_ms": start_ms,
        "end_ms": int(meta["end_ms"]),
        "url": f"https://www.youtube.com/watch?v={meta['video_id']}&t={linked_s}s",
        "label": _label(start_ms),
        "_text": doc.page_content,  # stripped before returning to the client
    }


def _label(ms: int) -> str:
    s = ms // 1000
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02}:{s % 60:02}"
    return f"{s // 60:02}:{s % 60:02}"


def build_context(docs: list[Document]) -> tuple[str, dict[str, dict]]:
    """Format chunks for the prompt with stable ids the model can cite."""
    blocks, cmap = [], {}
    for i, doc in enumerate(docs, start=1):
        cid = f"c{i}"
        cmap[cid] = to_citation(doc)
        blocks.append(
            f"[{cid}] {cmap[cid]['title']} @ {cmap[cid]['label']}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(blocks), cmap


# ---------------------------------------------------------------
# Verification
# ---------------------------------------------------------------

def _longest_run(quote_tokens: list[str], chunk_tokens: list[str]) -> int:
    """Longest contiguous shared token run.

    Set overlap is unusable here -- generic filler like "the upper body and a
    week" scores 1.00 against a chunk containing all those words separately.
    Contiguity is what distinguishes a real quote.
    """
    positions: dict[str, list[int]] = {}
    for i, tok in enumerate(chunk_tokens):
        positions.setdefault(tok, []).append(i)

    best = 0
    for qi in range(len(quote_tokens)):
        for start in positions.get(quote_tokens[qi], []):
            n = 0
            while (
                qi + n < len(quote_tokens)
                and start + n < len(chunk_tokens)
                and quote_tokens[qi + n] == chunk_tokens[start + n]
            ):
                n += 1
            if n > best:
                best = n
    return best


def verify_quote(quote: str, citation: dict) -> dict:
    """Check the model's quote occurs verbatim in the chunk it cited.

    Returns score in [0, 1] and, when verified, an approximate timestamp
    interpolated from position within the chunk. That interpolation is rough;
    a proper span table would make it exact.
    """
    qt = normalize(quote).split()
    ct = normalize(citation["_text"]).split()

    if not qt:
        return {"verified": False, "score": 0.0, "reason": "empty_quote"}

    run = _longest_run(qt, ct)
    score = run / len(qt)
    verified = score >= VERIFY_THRESHOLD

    result = {"verified": verified, "score": round(score, 3), "reason": "run_ratio"}

    if verified:
        joined = " ".join(ct)
        pos = joined.find(" ".join(qt))
        if pos >= 0:
            frac = pos / max(len(joined), 1)
            span = citation["end_ms"] - citation["start_ms"]
            result["approx_ms"] = citation["start_ms"] + int(frac * span)

    return result


# ---------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------

def resolve(answer: str, cmap: dict) -> tuple[list[dict], list[str]]:
    """Extract cited ids and quotes from the answer.

    Returns (resolved, hallucinated). Verification happens here rather than in
    the caller because it needs the chunk text, and _text must not leak past
    this module.

    A citation carrying a quote gains `verified`, `score` and (when verified)
    `approx_ms`. A bare [cN] with no quote gains `verified: None` -- meaning
    "not checkable", which is deliberately distinct from `False`, "checked and
    failed".
    """
    seen, resolved, bad = set(), [], []

    for cid, raw_quote in _CITE.findall(answer):
        quote = raw_quote.strip().strip(_QUOTE_CHARS).strip()

        # Dedupe on (id, quote), not id alone. A model routinely cites the same
        # block for two different claims -- [c4: "..."] ... [c4: "..."] -- and
        # collapsing those on id would drop the second quote BEFORE it is
        # verified, quietly shrinking the grounding guarantee to whichever
        # quote happened to come first.
        key = (cid, normalize(quote))
        if key in seen:
            continue
        seen.add(key)

        if cid not in cmap:
            if cid not in bad:
                bad.append(cid)
            continue

        entry = {k: v for k, v in cmap[cid].items() if k != "_text"}

        if quote:
            entry["quote"] = quote
            entry.update(verify_quote(quote, cmap[cid]))
        else:
            entry["verified"] = None

        resolved.append(entry)

    if bad:
        log.warning(f"Model emitted unknown citation ids: {bad}")

    return resolved, bad


def _combine_verified(a, b):
    """Conservative merge of two verification flags.

    False beats True beats None: one failed quote inside a merged span makes
    the whole span suspect, and "not checkable" never upgrades to "checked".
    """
    if a is False or b is False:
        return False
    if a is True or b is True:
        return True
    return None


def merge_adjacent(citations: list[dict], gap_ms: int = 1000) -> list[dict]:
    """Collapse contiguous or overlapping spans from the same video.

    With overlapping chunks this merges aggressively. That is usually what you
    want for display -- three links into the same 90 seconds is worse than one.

    Merging carries verification forward instead of dropping it. Each output
    citation exposes `quotes` (a list, possibly empty) and an aggregated
    `verified`; the singular `quote` from resolve() is folded into that list,
    because otherwise a merge would silently discard the evidence for every
    claim after the first.
    """
    if not citations:
        return []

    def _seed(c: dict) -> dict:
        out = dict(c)
        quote = out.pop("quote", None)
        out["quotes"] = [quote] if quote else []
        return out

    ordered = sorted(citations, key=lambda c: (c["video_id"], c["start_ms"]))
    merged = [_seed(ordered[0])]

    for c in ordered[1:]:
        last = merged[-1]
        if c["video_id"] == last["video_id"] and c["start_ms"] <= last["end_ms"] + gap_ms:
            last["end_ms"] = max(last["end_ms"], c["end_ms"])
            last["verified"] = _combine_verified(
                last.get("verified"), c.get("verified")
            )
            if c.get("quote"):
                last["quotes"].append(c["quote"])
            # Lowest score is the honest one for a merged span.
            if "score" in c:
                last["score"] = min(last.get("score", 1.0), c["score"])
        else:
            merged.append(_seed(c))

    return merged


def strip_answer(answer: str) -> str:
    """Remove citation markers for display when the UI renders links separately."""
    return _WS.sub(" ", _CITE.sub("", answer)).strip()