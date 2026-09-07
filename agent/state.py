from typing import Annotated, TypedDict, Literal

from langgraph.graph.message import add_messages


class QueryClassification(TypedDict):
    intent: Literal["weather", "report", "product", "complex"]
    # The query rewritten to be understandable without the conversation history
    # (pronouns and omissions resolved). Equals the raw query on the first turn.
    standalone_query: str
    # 0-1. Below the configured threshold the router escalates to the `complex`
    # fan-out rather than trusting a shaky single-path decision.
    confidence: float
    # "YYYY-MM" when the user named a month, empty string otherwise.
    month: str
    topic: str
    summary: str


def _add(a, b):
    return (a or 0) + (b or 0)


class GraphState(TypedDict):
    query: str
    ip: str
    # The signed-in user for this session. Session-scoped, so unlike the scratch
    # fields below it is NOT reset per turn.
    session_user_id: str
    # query after coreference resolution, used by downstream nodes
    standalone_query: str
    # output of nodes
    location: dict  # city, lat, lon
    weather: str
    user_id: str
    month: str
    external_data: str
    rag_result: str
    is_report: bool
    # append-only conversation history, persisted across turns by the checkpointer
    messages: Annotated[list, add_messages]
    final_response: str
    # classification result
    classification: QueryClassification
    # fan-in barrier
    expected_branches: int
    completed_branches: Annotated[int, _add]
