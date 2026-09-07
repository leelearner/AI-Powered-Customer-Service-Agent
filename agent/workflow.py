from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from agent.state import GraphState
from rag.rag_service import get_rag_service
from utils.logger_handler import logger
from agent.nodes import (
    intent_router,
    get_location,
    get_weather,
    get_user_id,
    get_month,
    fetch_data_node,
    rag_node,
    join_node,
    synthesize_node,
)


# Note on retries: LangGraph's node-level `retry_policy` wraps the node from the
# outside, but `_degrade_on_error` in nodes.py catches inside it — so a retry policy
# here would never fire. Transient-failure retry therefore lives one layer down, in
# the HTTP session in agent/tools/agent_tools.py; this layer only degrades.
def build_workflow(checkpointer=None, ingest: bool = True):
    """Build and compile the agent workflow graph.

    Args:
        checkpointer: conversation memory backend. Defaults to an in-process
            InMemorySaver, which keeps history for the lifetime of the object.
        ingest: load new knowledge documents into the vector store once at build
            time. Set False in tests, which stub the RAG layer out entirely.
    """
    if ingest:
        # Once per process, rather than on every RAG query as it used to be.
        try:
            get_rag_service().ingest()
        except Exception as e:
            logger.error(f"Knowledge ingestion failed, continuing without it: {e}")

    graph = StateGraph(GraphState)

    # Register all nodes
    graph.add_node("intent_router", intent_router)
    graph.add_node("get_location", get_location)
    graph.add_node("get_weather", get_weather)
    graph.add_node("get_user_id", get_user_id)
    graph.add_node("get_month", get_month)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("rag", rag_node)
    # defer=True acts as a join barrier: parallel branches reach `synthesize` at
    # different depths (rag at step 2, get_weather/fetch_data at step 3), so
    # without it the `complex` path runs synthesize twice — once with partial
    # context, then again with the full context.
    graph.add_node("synthesize", synthesize_node, defer=True)

    # Entry point
    graph.add_edge(START, "intent_router")

    # Terminal node
    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
