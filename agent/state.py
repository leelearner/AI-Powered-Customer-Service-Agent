from typing import Annotated, TypedDict, Literal


class QueryClassification(TypedDict):
    intent: Literal["weather", "report", "product", "complex"]
    topic: str
    summary: str


def _add(a, b):
    return (a or 0) + (b or 0)


class GraphState(TypedDict):
    query: str
    ip: str
    # output of nodes
    location: dict  # city, lat, lon
    weather: str
    user_id: str
    month: str
    external_data: str
    rag_result: str
    is_report: bool
    messages: list
    final_response: str
    # classification result
    classification: QueryClassification
    # fan-in barrier
    expected_branches: int
    completed_branches: Annotated[int, _add]
