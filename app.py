import time

import streamlit as st
from agent.workflow import build_workflow
from streamlit_js_eval import streamlit_js_eval

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

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("Agent is thinking..."):
        initial_state = {
            "query": prompt,
            "ip": st.session_state["metadata"].get("ip", ""),
        }

        result = st.session_state["workflow"].invoke(initial_state)
        final_response = result.get("final_response", "无法生成回答，请重试。")

        def stream_chars(text):
            for char in text:
                time.sleep(0.01)
                yield char

        st.chat_message("assistant").write_stream(stream_chars(final_response))
        st.session_state["message"].append(
            {"role": "assistant", "content": final_response}
        )
        st.rerun()
