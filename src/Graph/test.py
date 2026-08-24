"""End-to-end smoke test for the graph.

    python -m src.Graph.test

Run it as a module, not as a path: `python src/Graph/test.py` puts src/Graph on
sys.path instead of the repo root, and `src.*` stops resolving.

Scope is passed the way production will pass it: in config["configurable"],
because a thread is pinned to one video or one playlist for its whole life.
The URL form below is deliberate -- it proves _normalize_scope coerces it to a
bare video id. Passing a URL straight through to the vector filter matches zero
vectors and fails silently, which is the single easiest way to make working
retrieval look broken.
"""

import asyncio
import sys

from langchain_core.messages import HumanMessage

from src.graph.build_graph import get_graph

# LLM output routinely contains characters cp1252 cannot encode (narrow
# no-break space, curly quotes). Without this the console raises instead of
# printing the answer.
sys.stdout.reconfigure(encoding="utf-8")

# (question, scope, what it is probing)
CASES = [
    (
        "what is the latency to read from an SSD?",
        # URL form on purpose: proves _normalize_scope coerces it.
        ["https://youtu.be/FqR5vESuKe0?list=PLCRMIe5FDPsd0gVs500xeOewfySTsmEjf"],
        "covered -- expect a verified quote",
    ),
    (
        "how do I configure nginx worker processes?",
        ["FqR5vESuKe0"],
        "not covered -- expect a refusal",
    ),
    (
        "why do companies put kafka between their services?",
        ["7_wkWQ9rB5I"],
        "ASR writes Kafka as 'CFKA' here; a quote that silently fixes the "
        "spelling should fail verification",
    ),
    (
        "what delivery guarantees does kafka offer?",
        ["7_wkWQ9rB5I"],
        "three-part answer -- expect multiple citations",
    ),
]


async def ask(graph, question: str, scope: list[str]) -> None:
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"video_ids": scope}},
    )

    print(f"\nQ: {question}")
    print(f"   chunks retrieved : {len(state['chunks'])}")
    print(f"   grounded         : {state.get('grounded')}")
    print(f"   answer           : {state['messages'][-1].content.strip()}")

    for c in state.get("citations", []):
        mark = {True: "verified", False: "UNVERIFIED", None: "no quote"}[
            c.get("verified")
        ]
        print(f"   cite             : {c['label']}  [{mark}]  {c['url']}")
        for q in c.get("quotes", []):
            print(f"                      “{q}”")


async def main() -> None:
    graph = get_graph()
    for question, scope, probing in CASES:
        print(f"\n--- {probing}")
        await ask(graph, question, scope)


if __name__ == "__main__":
    asyncio.run(main())
