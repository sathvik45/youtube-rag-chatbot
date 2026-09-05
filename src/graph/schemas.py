from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """What flows between nodes.

    Everything here is checkpointed on every turn, so it holds only what a
    later node actually needs. Thread scope (video_ids) is deliberately NOT
    here -- one thread is pinned to one video or one playlist for its whole
    life, so it belongs in config["configurable"], not in per-turn state.

    chunks are plain dicts rather than Documents because state gets serialised
    by the checkpointer; dicts survive that and schema changes without a
    pydantic model in the way.
    """

    messages: Annotated[List[BaseMessage], add_messages]

    # Standalone, pronoun-resolved rewrite of the latest message. What we
    # actually search with -- the raw message may be "what about the second one?"
    query: str

    # Retrieved chunk metadata + text. Empty means we found nothing and the
    # graph must refuse rather than generate.
    chunks: List[dict]

    # Exactly what the generator was shown, with [c1]/[c2] ids. Kept for
    # debugging and so the citation step can rebuild the same id mapping.
    context: str

    # Resolved, de-duplicated citations for the client. Filled after the
    # answer, from the ids the model actually emitted. Each carries `quotes`
    # and a `verified` flag.
    citations: List[dict]

    # Did every quote the model offered actually appear in the chunk it cited?
    # True  = every quoted claim checked out
    # False = at least one quote is not in its chunk -- treat the answer as
    #         ungrounded regardless of how confident it reads
    # None  = nothing was checkable (no quotes emitted, or nothing retrieved)
    grounded: bool | None

    # An explicit terminal outcome lets the API persist the correct message
    # status without trying to infer intent from user-facing answer text.
    outcome: Literal["answered", "refused_no_context"]
