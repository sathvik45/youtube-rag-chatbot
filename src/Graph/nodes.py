"""Graph nodes.

Shape of the pipeline:

    rewrite -> retrieve -> (context?) --no--> refuse -> END
                              |yes
                              v
                          generate -> cite -> END

The retrieve node is deliberately thin. Every retrieval knob (k, per-video cap,
score threshold) lives in rag.retriever.retrieve and core.config, because
evals/scorers/run_retrieval_eval.py calls retrieve() directly. Policy added
here instead of there would mean the golden set measures something production
does not do.
"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from src.core.config import settings
from src.core.logging import get_logger
from src.graph.prompts import Generate_prompt, Query_rewrite_prompt
from src.graph.schemas import State
from src.llm.client import get_llm
from src.rag.Citations import build_context, merge_adjacent, resolve
from src.rag.fetch_transcripts import extract_video_id
from src.rag.retriever import retrieve

log = get_logger(__name__)

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything about that in this video. "
    "Try rephrasing, or ask about something the video actually covers."
)


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

def _format_window(messages: list[BaseMessage], max_chars: int = 500) -> str:
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = m.content.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        lines.append(f"{role} : {text}")
    return "\n".join(lines)


def _normalize_scope(video_ids: list[str]) -> list[str]:
    """Coerce whatever the caller passed into bare 11-char video ids.

    Chunk metadata stores the bare id, so a full watch URL in the scope filter
    matches zero vectors and fails SILENTLY -- retrieval returns nothing and it
    looks like a retrieval quality problem. Normalising here turns that into a
    loud error at the boundary instead.

    Reuses extract_video_id from the ingest path so URL parsing has exactly one
    implementation.
    """
    if not video_ids:
        raise ValueError("Thread has no video scope")

    out = []
    for raw in video_ids:
        try:
            out.append(extract_video_id(raw))
        except Exception as e:
            raise ValueError(
                f"Cannot read a video id from scope entry {raw!r}. A playlist "
                f"thread needs its expanded video ids, not the playlist URL."
            ) from e
    return out


def _to_documents(chunks: list[dict]) -> list[Document]:
    """Rebuild Documents for Citations.build_context.

    State holds dicts (serialisable); build_context wants Documents. Converting
    here keeps Citations.py untouched and keeps the checkpointed state plain.
    """
    return [
        Document(
            page_content=c.get("text", ""),
            metadata={k: v for k, v in c.items() if k != "text"},
        )
        for c in chunks
    ]


# ---------------------------------------------------------------
# nodes
# ---------------------------------------------------------------

async def rewrite_node(state: State) -> dict:
    """Turn the latest message into a standalone search query.

    Retrieval is a similarity search with no memory, so "what about the second
    one?" retrieves nothing useful. The rewrite resolves pronouns against the
    recent window. On the first turn there is nothing to resolve, so we skip
    the LLM call entirely.
    """
    history = state["messages"][:-1]
    latest = state["messages"][-1].content

    if not history:
        return {"query": latest}

    window = _format_window(history[-6:])
    chain = Query_rewrite_prompt | get_llm() | StrOutputParser()
    raw = await chain.ainvoke({"history": window, "question": latest})
    query = raw.strip().strip('"')

    # A rewrite that ran away is worse than no rewrite: it drags retrieval
    # off-topic. Fall back to the raw message rather than trusting it.
    if not query or len(query) > 4 * len(latest) + 200:
        log.warning("Discarding suspicious rewrite, using raw message")
        return {"query": latest}

    log.info(f"rewrote: {latest!r} -> {query!r}")
    return {"query": query}


def retrieve_node(state: State, config: RunnableConfig) -> dict:
    """Scoped similarity search. Thin by design -- see module docstring.

    Sync, not async, on purpose: retrieve() blocks twice (local embedding on
    CPU, then the Pinecone call). Awaiting it inside an async node would block
    the event loop for every concurrent request. LangGraph runs sync nodes in a
    threadpool, which is exactly what this wants.
    """
    scope = _normalize_scope(config["configurable"]["video_ids"])

    docs = retrieve(
        state["query"],
        scope,
        k=settings.retrive_K,
        # No-op when the thread scopes a single video; retrieve() checks.
        per_video_cap=settings.per_video_cap,
        score_threshold=settings.score_threshold,
    )

    log.info(f"retrieved {len(docs)} chunks over {len(scope)} video(s)")
    if not docs:
        log.warning(
            f"no chunks for {state['query']!r} "
            f"(score_threshold={settings.score_threshold})"
        )

    return {
        "chunks": [{**d.metadata, "text": d.page_content} for d in docs]
    }


def has_context(state: State) -> str:
    """Router. This is where 'answer only from retrieved context' is actually
    enforced -- structurally, by never reaching the generator without chunks.

    A prompt rule asking the model to refuse is a request. An edge that does
    not exist is a guarantee.
    """
    return "generate" if state["chunks"] else "refuse"


def refuse_node(state: State) -> dict:
    """Nothing retrieved, so there is nothing to ground an answer in.

    No LLM call: any model handed an empty context and a question will produce
    something, and that something is exactly the failure mode we are avoiding.
    """
    log.info("refusing: no context")
    return {
        "messages": [AIMessage(content=NO_CONTEXT_MESSAGE)],
        "context": "",
        "citations": [],
        "grounded": None,
        "outcome": "refused_no_context",
    }


async def Generate_node(state: State) -> dict:
    """Answer from context, with citation ids the model can point at.

    Uses state["query"] rather than the raw last message: the rewrite is what
    retrieval matched, so the question the model answers should be the same
    question the context was fetched for.
    """
    context, _ = build_context(_to_documents(state["chunks"]))

    chain = Generate_prompt | get_llm()
    answer = await chain.ainvoke(
        {"context": context, "question": state["query"]}
    )

    return {"messages": [answer], "context": context}


def cite_node(state: State) -> dict:
    """Resolve [cN] ids into linkable timestamps and check the quotes.

    build_context is deterministic given the chunk order in state, so
    rebuilding the id map here yields the same ids the generator saw, without
    checkpointing a second copy of every chunk's text.

    Two independent failure signals come out of this:

      hallucinated ids -- the model cited a block that was never in context
      failed quotes    -- the model cited a real block but the words it
                          attributed to that block are not in it

    The second is the one that matters for "answer only from retrieved
    context", and it is arithmetic rather than a judgement call: longest
    contiguous token run against the cited chunk, threshold 0.9. Paraphrase
    scores around 0.36 and fabrication around 0.22, so the gap is wide.

    Policy note: an unverified answer is currently ANNOTATED, not blocked.
    state["grounded"] carries the verdict so the API layer can decide. Blocking
    here would mean returning nothing on questions the retriever answered
    correctly but the model quoted loosely, and you want to see the real
    failure rate before paying that price.
    """
    if not state["chunks"]:
        return {"citations": [], "grounded": None}

    _, cmap = build_context(_to_documents(state["chunks"]))
    answer = state["messages"][-1].content

    resolved, hallucinated = resolve(answer, cmap)
    if hallucinated:
        log.warning(f"model cited unknown ids: {hallucinated}")

    checked = [c for c in resolved if c.get("verified") is not None]
    failed = [c for c in checked if c["verified"] is False]

    for c in failed:
        log.warning(
            f"unverified quote (score={c.get('score')}) for {c['video_id']} "
            f"@{c['label']}: {c.get('quote', '')[:80]!r}"
        )

    # None when nothing was checkable -- distinct from False, which means
    # something was checked and did not hold up.
    grounded = None if not checked else not failed

    citations = merge_adjacent(resolved)
    log.info(
        f"{len(citations)} citation(s), "
        f"{len(checked) - len(failed)}/{len(checked)} quotes verified"
    )
    return {
        "citations": citations,
        "grounded": grounded,
        "outcome": "answered",
    }
