from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from utils.prompt_loader import load_report_prompt, load_system_prompt
from langchain.tools.tool_node import ToolCallRequest
from typing import Any, Callable
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from langchain.agents import AgentState
from langgraph.runtime import Runtime


@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    logger.info(f"[tool monitor] Executing tool: {request.tool_call['name']}")
    logger.info(f"[tool monitor] Input Args: {request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(
            f"[tool monitor] Tool {request.tool_call['name']} executed successfully."
        )
        if request.tool_call["name"] == "fill_context_for_report":
            if isinstance(request.runtime.context, dict):
                request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(
            f"[tool monitor] Tool {request.tool_call['name']} execution failed: {e}"
        )
        raise e


@before_model
def log_before_model(
    state: AgentState,
    runtime: Runtime,
):
    logger.info(
        f"[log_before_model] Agent is about to call the model with {len(state['messages'])} messages."
    )
    logger.info(
        f"[log_before_model]{type(state['messages'][-1]).__name__} content: {state['messages'][-1].content}"
    )

    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest):
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()

    return load_system_prompt()
