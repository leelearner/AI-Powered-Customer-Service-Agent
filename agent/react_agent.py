from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)

from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[
                rag_summarize,
                get_weather,
                get_user_location,
                get_user_id,
                get_current_month,
                fetch_external_data,
                fill_context_for_report,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        for chunk in self.agent.stream(
            input_dict, stream_mode="values", context={"report": False}
        ):

            latest_message = chunk["messages"][-1]
            if isinstance(latest_message.content, str):
                text = latest_message.content.strip()
            elif isinstance(latest_message.content, list) and latest_message.content:
                text = latest_message.content[0]["text"].strip()
            else:
                text = ""

            yield text + "\n"


if __name__ == "__main__":
    agent = ReactAgent()
    query = "扫地机器人在我所在地区的气温下如何保养"
    for response in agent.execute_stream(query):
        print(response, end="", flush=True)
