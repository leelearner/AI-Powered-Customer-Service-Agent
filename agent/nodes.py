from functools import wraps

from agent.tools.agent_tools import (
    rag_summarize,
    get_weather_tool,
    get_user_location,
    fetch_external_data,
    fill_context_for_report,
    available_user_ids,
    available_months,
)
from agent.state import GraphState, QueryClassification
from chat.model.factory import classification_model, synthesis_model
from langgraph.types import Command
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from utils.config_handler import agent_conf
from utils.prompt_loader import (
    load_classification_prompt,
    load_system_prompt,
    load_report_prompt,
)
from utils.logger_handler import logger


def _degrade_on_error(goto: str, **fallback):
    """Turn a node failure into an empty result that still advances the graph.

    A customer-service agent should answer with partial context rather than hand
    the user a stack trace. `synthesize_node` already assembles its background
    section from whichever fields are non-empty, so a degraded node simply drops
    out of that section.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(state: GraphState) -> Command:
            try:
                return func(state)
            except Exception as e:
                logger.error(f"Node '{func.__name__}' failed, degrading: {e}")
                return Command(update=dict(fallback), goto=goto)

        return wrapper

    return decorator


def _recent_history(state: GraphState) -> list:
    """Trailing conversation messages from previous turns, oldest first.

    Two constraints, both of which Anthropic rejects with a 400 if violated:
    - Empty-text messages are dropped. A message carrying only an empty text block
      makes every later turn in the thread fail with "text content blocks must be
      non-empty", so filtering here lets an already-poisoned session recover.
    - The window is trimmed to start on a HumanMessage, since the turn after the
      system prompt must be a user turn.
    """
    limit = agent_conf.get("history_max_messages", 6)
    if limit <= 0:
        return []
    history = [m for m in (state.get("messages") or []) if m.text.strip()]
    window = history[-limit:]
    while window and not isinstance(window[0], HumanMessage):
        window.pop(0)
    return window


def _format_history(messages: list) -> str:
    """Render prior messages as plain text for the classification prompt."""
    if not messages:
        return "（无历史，这是本次会话的第一轮提问）"
    return "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '客服'}: {m.text}"
        for m in messages
    )


def intent_router(
    state: GraphState,
) -> Command:
    """Classify intent and rewrite the query into a standalone question, then route.

    Coreference resolution is folded into this existing structured-output call so
    multi-turn support costs no extra LLM round trip.
    """

    structured_llm = classification_model.with_structured_output(QueryClassification)
    classification_prompt = load_classification_prompt().format(
        input=state["query"],
        history=_format_history(_recent_history(state)),
    )

    # The classification call itself can fail (timeout, rate limit, bad credentials).
    # Unhandled, that takes down the whole graph on a single flaky request.
    try:
        classification = structured_llm.invoke(classification_prompt)
    except Exception as e:
        logger.error(f"Intent classification call failed, defaulting to rag: {e}")
        classification = None

    if isinstance(classification, dict):
        standalone_query = classification.get("standalone_query") or state["query"]
        intent = classification.get("intent")
        confidence = classification.get("confidence")
        threshold = agent_conf.get("classification_confidence_threshold", 0.6)

        if isinstance(confidence, (int, float)) and confidence < threshold:
            # Uncertain classification: gather everything rather than commit to one
            # path. Costs an extra call or two; beats answering the wrong question.
            logger.warning(
                f"Low classification confidence ({confidence} < {threshold}) for "
                f"intent '{intent}', escalating to complex fan-out."
            )
            intent = "complex"

        if intent == "weather":
            goto = "get_location"
        elif intent == "report":
            goto = ["get_user_id", "get_month"]
        elif intent == "product":
            goto = "rag"
            expected = 1
        else:
            goto = ["get_location", "rag", "get_user_id", "get_month"]
            expected = 3  # get_weather→join, rag→join, fetch_data(merged)→join
    else:
        logger.warning(
            f"LLM returned unexpected classification format: {classification}"
        )
        classification = {}
        standalone_query = state["query"]
        goto = "rag"
        expected = 1

    logger.info(
        f"Query classified as intent: {goto}, standalone_query='{standalone_query}', "
        f"confidence: {classification.get('confidence', 'n/a')}, "
        f"month: '{classification.get('month', '')}', "
        f"topic: {classification.get('topic', '')}"
    )

    # Reset this turn's scratch fields. The checkpointer persists state across
    # turns, so without an explicit resea the previous turn's weather/report data
    # would leak into this turn's synthesis context.
    # `session_user_id` is deliberately absent: it is session-scoped, not per-turn.
    return Command(
        update={
            "classification": classification,
            "standalone_query": standalone_query,
            "location": {},
            "weather": "",
            "user_id": "",
            "month": "",
            "external_data": "",
            "rag_result": "",
            "is_report": False,
        },
        goto=goto,
    )


@_degrade_on_error(goto="get_weather", location={})
def get_location(state: GraphState) -> Command:
    """Node to get user location based on IP."""
    location = get_user_location.invoke({"ip": state["ip"]})
    return Command(update={"location": location}, goto="get_weather")


@_degrade_on_error(goto="synthesize", weather="")
def get_weather(state: GraphState) -> Command:
    """Node to get weather info based on location."""
    location = state.get("location") or {}
    # Upstream may have degraded to an empty dict, so read defensively.
    city = location.get("city", "")
    lat, lon = location.get("lat", ""), location.get("lon", "")
    if not city or lat == "" or lon == "":
        logger.warning("Location unavailable, skipping weather lookup.")
        return Command(update={"weather": ""}, goto="synthesize")

    weather = get_weather_tool.invoke(
        {"city": city, "lat": str(lat), "lon": str(lon)}
    )
    return Command(update={"weather": weather}, goto="join")


@_degrade_on_error(goto="fetch_data", user_id="")
def get_user_id(state: GraphState) -> Command:
    """Node resolving the user whose records to read.

    Reads the session's selected user rather than picking one at random, and
    validates it against the ids actually present in the data.
    """
    known = available_user_ids()
    user_id = state.get("session_user_id", "")
    if user_id not in known:
        fallback = known[0] if known else ""
        if user_id:
            logger.warning(f"Unknown session user_id '{user_id}', using '{fallback}'.")
        user_id = fallback
    return Command(update={"user_id": user_id}, goto="fetch_data")


@_degrade_on_error(goto="fetch_data", month="")
def get_month(state: GraphState) -> Command:
    """Node resolving which month to report on.

    Prefers the month the classifier extracted from the query; falls back to the
    most recent month present in the data.
    """
    month = (state.get("classification") or {}).get("month", "")
    known = available_months()
    if month not in known:
        fallback = known[-1] if known else ""
        if month:
            logger.warning(f"Month '{month}' not present in data, using '{fallback}'.")
        month = fallback
    return Command(update={"month": month}, goto="fetch_data")


@_degrade_on_error(goto="synthesize", external_data="", is_report=False)
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
        goto="join",
    )


@_degrade_on_error(goto="synthesize", rag_result="")
def rag_node(state: GraphState) -> Command:
    """Node to retrieve relevant knowledge via RAG.

    Retrieval uses the rewritten standalone query: a follow-up like "那滤网呢？"
    carries no retrievable terms on its own.
    """
    query = state.get("standalone_query") or state["query"]
    result = rag_summarize.invoke({"query": query})
    logger.info(f"RAG retrieval completed for query: {query}")
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
        *_recent_history(state),
        HumanMessage(content=user_message),
    ]

    # Stream rather than invoke: LangGraph's stream_mode="messages" only emits
    # tokens as they arrive if the underlying call actually streams. The chunks
    # are accumulated back into a single message for the state update.
    response = None
    try:
        for chunk in synthesis_model.stream(messages):
            response = chunk if response is None else response + chunk
    except Exception as e:
        # app.py renders a retry message when final_response is empty.
        logger.error(f"Synthesis call failed: {e}")
        response = None

    # `.text` concatenates only the text blocks, skipping Anthropic thinking blocks
    final_response = response.text if response is not None else ""
    logger.info("Final response generated.")

    # `messages` uses the add_messages reducer, so return only this turn's delta.
    #
    # Store plain text, never the accumulated chunk. The chunk carries Anthropic's
    # block structure — including thinking blocks, which are empty under the default
    # display="omitted" — and replaying a block whose text is empty makes the next
    # turn fail with "text content blocks must be non-empty". Note that `if response`
    # would NOT catch that case: a message object is always truthy, so the emptiness
    # has to be tested on the text.
    #
    # The raw query is stored (not the context-augmented `user_message`) so the next
    # turn's rewrite sees the real conversation. A turn that produced no text is not
    # recorded at all, keeping history strictly user/assistant alternating.
    turn = (
        [HumanMessage(content=state["query"]), AIMessage(content=final_response)]
        if final_response.strip()
        else []
    )

    return {
        "final_response": final_response,
        "messages": turn,
    }
