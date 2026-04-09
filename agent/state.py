from typing import TypedDict, Literal


class QueryClassification(TypedDict):
    intent: Literal["weather", "report", "product", "complex"]
    topic: str
    summary: str


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
