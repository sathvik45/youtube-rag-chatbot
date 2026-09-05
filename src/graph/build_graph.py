"""Graph wiring.

    START -> rewrite -> retrieve -> (context?) --no--> refuse --> END
                                        |yes
                                        v
                                    generate -> cite -> END

The conditional edge out of retrieve is the load-bearing part. "Answer only
from retrieved context" is enforced by the graph's shape, not by asking the
model nicely: with no chunks there is no path to the generator at all.
"""

from langgraph.graph import END, START, StateGraph

from src.graph import nodes
from src.graph.schemas import State


def build_graph():
    g = StateGraph(State)

    g.add_node("rewrite", nodes.rewrite_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("generate", nodes.Generate_node)
    g.add_node("cite", nodes.cite_node)
    g.add_node("refuse", nodes.refuse_node)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")

    g.add_conditional_edges(
        "retrieve",
        nodes.has_context,
        {"generate": "generate", "refuse": "refuse"},
    )

    g.add_edge("generate", "cite")
    g.add_edge("cite", END)
    g.add_edge("refuse", END)

    return g.compile()


def get_graph():
    return build_graph()