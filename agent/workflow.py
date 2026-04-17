from langgraph.graph import StateGraph, START, END
from agent.state import GraphState
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


def build_workflow():
    """Build and compile the agent workflow graph."""
    graph = StateGraph(GraphState)

    # Register all nodes
    graph.add_node("intent_router", intent_router)
    graph.add_node("get_location", get_location)
    graph.add_node("get_weather", get_weather)
    graph.add_node("get_user_id", get_user_id)
    graph.add_node("get_month", get_month)
    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("rag", rag_node)
    graph.add_node("join", join_node)
    graph.add_node("synthesize", synthesize_node)

    # Entry point
    graph.add_edge(START, "intent_router")

    # Terminal node
    graph.add_edge("synthesize", END)

    return graph.compile()
