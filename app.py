import time

import streamlit as st
from agent.react_agent import ReactAgent
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

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []
    with st.spinner("Agent is thinking..."):
        res_strem = st.session_state["agent"].execute_stream(
            prompt, st.session_state["metadata"]
        )

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                for char in chunk:
                    time.sleep(0.01)  # Simulate delay for streaming effect
                    yield char

        st.chat_message("assistant").write_stream(capture(res_strem, response_messages))
        st.session_state["message"].append(
            {"role": "assistant", "content": response_messages[-1]}
        )
        st.rerun()
