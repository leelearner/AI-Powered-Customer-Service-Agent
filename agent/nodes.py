from typing import Literal

from agent.tools.agent_tools import (
    rag_summarize,
    get_weather_tool,
    get_user_location,
    get_user_id_tool,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)
from agent.state import GraphState, QueryClassification
from langgraph.types import Command
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from utils.prompt_loader import (
    load_classification_prompt,
    load_system_prompt,
    load_report_prompt,
)
from utils.logger_handler import logger

llm = ChatAnthropic(model_name="claude-sonnet-4-6", timeout=120, stop=None)


def intent_router(
    state: GraphState,
) -> Command:
    """Use LLM to classify query intent, then route accordingly."""

    structured_llm = llm.with_structured_output(QueryClassification)
    classification_prompt = load_classification_prompt().format(input=state["query"])
    classification = structured_llm.invoke(classification_prompt)

    if isinstance(classification, dict):
        if classification["intent"] == "weather":
            goto = "get_location"
        elif classification["intent"] == "report":
            goto = ["get_user_id", "get_month"]
        elif classification["intent"] == "product":
            goto = "rag"
        else:
            goto = ["get_location", "rag", "get_user_id", "get_month"]
    else:
        logger.warning(
            f"LLM returned unexpected classification format: {classification}"
        )
        goto = "rag"

    logger.info(
        f"Query classified as intent: {goto} with topic: {classification.get('topic', '')}"
    )

    return Command(update={"classification": classification}, goto=goto)


def get_location(state: GraphState) -> Command:
    """Node to get user location based on IP."""
    location = get_user_location.invoke({"ip": state["ip"]})
    return Command(update={"location": location}, goto="get_weather")


def get_weather(state: GraphState) -> Command:
    """Node to get weather info based on location."""
    location = state["location"]
    weather = get_weather_tool.invoke(
        {
            "city": location["city"],
            "lat": str(location["lat"]),
            "lon": str(location["lon"]),
        }
    )
    return Command(update={"weather": weather}, goto="synthesize")


def get_user_id(state: GraphState) -> Command:
    """Node to get user id."""
    user_id = get_user_id_tool.invoke({})
    return Command(update={"user_id": user_id}, goto="fetch_data")


def get_month(state: GraphState) -> Command:
    """Node to get current month."""
    month = get_current_month.invoke({})
    return Command(update={"month": month}, goto="fetch_data")


def fetch_data_node(state: GraphState) -> Command:
    """Node to fetch external user data and set report context.

    Equivalent to fetch_external_data + fill_context_for_report tools,
    plus the report_prompt_switch middleware trigger.
    """
    user_id = state.get("user_id", "")
    month = state.get("month", "")
    is_report = state.get("classification", {}).get("intent") == "report"

    external = ""
    if user_id and month:
        external = fetch_external_data.invoke({"user_id": user_id, "month": month})
        logger.info(f"Fetched external data for user_id={user_id}, month={month}")
        if is_report:
            fill_context_for_report.invoke({})

    return Command(
        update={"external_data": external, "is_report": is_report},
        goto="synthesize",
    )


def rag_node(state: GraphState) -> Command:
    """Node to retrieve relevant knowledge via RAG."""
    result = rag_summarize.invoke({"query": state["query"]})
    logger.info(f"RAG retrieval completed for query: {state['query']}")
    return Command(update={"rag_result": result}, goto="synthesize")


def synthesize_node(state: GraphState) -> dict:
    """Node to synthesize the final response from all collected context.

    Equivalent to log_before_model middleware + report_prompt_switch middleware.
    """
    # log_before_model middleware: log before calling LLM
    logger.info(
        f"Synthesizing final response, query='{state['query']}', is_report={state.get('is_report')}"
    )

    # report_prompt_switch middleware: select prompt based on report flag
    if state.get("is_report"):
        system_prompt = load_report_prompt()
    else:
        system_prompt = load_system_prompt()

    # Assemble context gathered from upstream nodes
    context_parts = []
    if state.get("weather"):
        context_parts.append(f"天气信息: {state['weather']}")
    if state.get("external_data"):
        context_parts.append(f"用户数据: {state['external_data']}")
    if state.get("rag_result"):
        context_parts.append(f"知识库检索结果: {state['rag_result']}")

    user_message = state["query"]
    if context_parts:
        context_str = "\n".join(context_parts)
        user_message = f"背景信息:\n{context_str}\n\n用户问题: {state['query']}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    final_response = response.content
    logger.info("Final response generated.")

    return {
        "final_response": final_response,
        "messages": (state.get("messages") or []) + [response],
    }
