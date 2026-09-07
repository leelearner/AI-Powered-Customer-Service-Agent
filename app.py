import uuid

import streamlit as st
from agent.tools.agent_tools import available_user_ids
from agent.workflow import build_workflow
from langchain_core.messages import AIMessageChunk
from streamlit_js_eval import streamlit_js_eval

# Initialise once per browser session. Reassigning this dict on every rerun would
# wipe the cached IP and re-trigger the front-end lookup on every single turn.
if "metadata" not in st.session_state:
    st.session_state["metadata"] = {}

if "ip" not in st.session_state["metadata"]:
    ip = streamlit_js_eval(
        js_expressions="fetch('https://api.ipify.org?format=json').then(res => res.json()).then(data => data.ip)",
        want_output=True,
        key="get_ip",
    )
    st.session_state["metadata"]["ip"] = ip

st.title("Agent customer service")
st.divider()

if "workflow" not in st.session_state:
    st.session_state["workflow"] = build_workflow()

# Stable conversation key: the checkpointer stores this session's history under it.
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "message" not in st.session_state:
    st.session_state["message"] = []

# Which customer we are acting for. Report data is per-user, so this has to be an
# explicit choice — it used to be re-randomised on every single turn.
user_id = st.sidebar.selectbox(
    "当前用户",
    options=available_user_ids(),
    key="session_user_id",
    help="报告类问题会读取该用户的使用记录。",
)

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    initial_state = {
        "query": prompt,
        "ip": st.session_state["metadata"].get("ip", ""),
        "session_user_id": user_id,
    }
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown("_Agent is thinking..._")

        def token_stream():
            """Yield the final answer token by token as the graph produces it.

            Both filters are required: `langgraph_node` suppresses tokens from the
            intent classifier and the RAG summariser, and the AIMessageChunk check
            suppresses the HumanMessage that synthesize returns in its state update.
            """
            first = True
            for chunk, meta in st.session_state["workflow"].stream(
                initial_state, config=config, stream_mode="messages"
            ):
                if meta.get("langgraph_node") != "synthesize":
                    continue
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = chunk.text
                if not text:
                    continue
                if first:
                    status.empty()
                    first = False
                yield text

        final_response = st.write_stream(token_stream())

    if not final_response:
        final_response = "无法生成回答，请重试。"
        status.markdown(final_response)

    st.session_state["message"].append(
        {"role": "assistant", "content": final_response}
    )
