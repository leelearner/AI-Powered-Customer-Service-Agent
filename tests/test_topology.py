"""Graph topology regression tests.

Runs the real compiled workflow with the LLM and the business tools stubbed out,
so it needs no API keys and makes no network calls.

    python tests/test_topology.py

Covers the three things that are easy to regress:
  1. `synthesize` must execute exactly once on every intent path (the `complex`
     path used to run it twice, wasting an LLM call on partial context).
  2. Scratch fields must be cleared between turns, or the checkpointer leaks the
     previous turn's weather/report data into the next answer.
  3. Conversation history must accumulate and reach the classifier.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model clients are constructed at import time; give them a dummy key so the
# import succeeds without real credentials.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

import requests  # noqa: E402
from langchain_core.messages import AIMessageChunk, HumanMessage  # noqa: E402

import agent.nodes as nodes  # noqa: E402
from agent.tools.agent_tools import available_months, available_user_ids  # noqa: E402
from agent.workflow import build_workflow  # noqa: E402


class _FakeTool:
    """Stands in for a @tool object: only `.invoke()` is ever called on it."""

    def __init__(self, value):
        self.value = value

    def invoke(self, args=None):
        return self.value


class _FakeStructured:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, prompt):
        self.llm.classification_prompts.append(prompt)
        return self.llm.classification


class FakeLLM:
    def __init__(self):
        self.classification = {}
        self.classification_prompts = []
        self.synthesize_calls = 0
        self.synthesize_messages = []

    def with_structured_output(self, schema):
        return _FakeStructured(self)

    def stream(self, messages):
        self.synthesize_calls += 1
        self.synthesize_messages.append(messages)
        yield AIMessageChunk(content="好的，")
        yield AIMessageChunk(content="这是回答。")


def install_stubs():
    """Replace the module-level models and tools in agent.nodes with fakes.

    Both models must be patched: since the classifier and the synthesiser are
    separate clients, missing one would silently call the real API.
    """
    fake = FakeLLM()
    nodes.classification_model = fake
    nodes.synthesis_model = fake
    nodes.get_user_location = _FakeTool(
        {"city": "Shanghai", "lat": 31.23, "lon": 121.47}
    )
    nodes.get_weather_tool = _FakeTool("晴，25°C，湿度 60%")
    nodes.fetch_external_data = _FakeTool("覆盖率:85% 主刷剩余60天")
    nodes.fill_context_for_report = _FakeTool("")
    nodes.rag_summarize = _FakeTool("滚刷建议每周清洁一次。")
    # Identity now comes from session state and the classifier, backed by the real
    # CSV — left unstubbed so the tests exercise the actual lookup.
    return fake


def human_text(messages):
    """The final HumanMessage handed to the LLM by synthesize."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def run(
    app,
    fake,
    query,
    intent,
    thread_id,
    standalone=None,
    confidence=0.9,
    month="2025-06",
    user_id="1001",
):
    fake.classification = {
        "intent": intent,
        "standalone_query": standalone or query,
        "confidence": confidence,
        "month": month,
        "topic": "测试",
        "summary": "测试",
    }
    return app.invoke(
        {"query": query, "ip": "1.2.3.4", "session_user_id": user_id},
        config={"configurable": {"thread_id": thread_id}},
    )


failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}{(' -- ' + detail) if detail else ''}")
        failures.append(name)


def main():
    print("\n[1] synthesize runs exactly once per intent, with complete context")
    expected_context = {
        "product": ["知识库检索结果"],
        "report": ["用户数据"],
        "weather": ["天气信息"],
        "complex": ["天气信息", "用户数据", "知识库检索结果"],
    }
    for intent, expected in expected_context.items():
        fake = install_stubs()
        app = build_workflow(ingest=False)
        result = run(app, fake, f"测试{intent}问题", intent, f"t-{intent}")
        context = human_text(fake.synthesize_messages[-1])
        check(
            f"{intent}: synthesize called once",
            fake.synthesize_calls == 1,
            f"called {fake.synthesize_calls}x",
        )
        check(
            f"{intent}: context complete {expected}",
            all(marker in context for marker in expected),
            context[:120],
        )
        check(
            f"{intent}: final_response populated",
            result.get("final_response") == "好的，这是回答。",
            repr(result.get("final_response")),
        )

    print("\n[2] scratch fields are cleared between turns")
    fake = install_stubs()
    app = build_workflow(ingest=False)
    run(app, fake, "今天天气适合用扫地机器人吗", "weather", "t-leak")
    turn1 = human_text(fake.synthesize_messages[-1])
    run(app, fake, "滚刷多久清洁一次", "product", "t-leak")
    turn2 = human_text(fake.synthesize_messages[-1])
    check("turn 1 sees weather", "天气信息" in turn1, turn1[:120])
    check("turn 2 does NOT see stale weather", "天气信息" not in turn2, turn2[:160])
    check("turn 2 sees rag result", "知识库检索结果" in turn2, turn2[:160])

    print("\n[3] history accumulates and reaches the classifier")
    fake = install_stubs()
    app = build_workflow(ingest=False)
    run(app, fake, "滚刷多久清洁一次？", "product", "t-hist")
    run(app, fake, "那滤网呢？", "product", "t-hist", standalone="滤网多久清洁一次？")
    state = app.get_state({"configurable": {"thread_id": "t-hist"}})
    stored = state.values["messages"]
    check("4 messages stored after 2 turns", len(stored) == 4, f"got {len(stored)}")
    check(
        "stored human message is the raw query",
        stored[2].content == "那滤网呢？",
        repr(stored[2].content),
    )
    check(
        "turn 2 classification prompt carries turn 1",
        "滚刷多久清洁一次？" in fake.classification_prompts[-1],
    )
    check(
        "turn 2 synthesize receives prior turn as history",
        any(
            isinstance(m, HumanMessage) and m.content == "滚刷多久清洁一次？"
            for m in fake.synthesize_messages[-1]
        ),
    )

    print("\n[4] rag retrieval uses the rewritten standalone query")
    captured = {}

    class _RecordingRag:
        def invoke(self, args):
            captured["query"] = args["query"]
            return "检索结果"

    fake = install_stubs()
    nodes.rag_summarize = _RecordingRag()
    app = build_workflow(ingest=False)
    run(app, fake, "那滤网呢？", "product", "t-rewrite", standalone="滤网多久清洁一次？")
    check(
        "rag queried with standalone_query",
        captured.get("query") == "滤网多久清洁一次？",
        repr(captured.get("query")),
    )

    print("\n[5] node failures degrade instead of crashing the graph")

    class _Exploding:
        def invoke(self, args=None):
            raise requests.RequestException("simulated outage")

    fake = install_stubs()
    nodes.get_user_location = _Exploding()
    nodes.rag_summarize = _Exploding()
    nodes.fetch_external_data = _Exploding()
    app = build_workflow(ingest=False)
    try:
        result = run(app, fake, "综合问题", "complex", "t-degrade")
        crashed = False
    except Exception as e:  # noqa: BLE001 - the point is that nothing escapes
        result, crashed = {}, repr(e)
    check("graph completes despite every external call failing", crashed is False, str(crashed))
    check("synthesize still ran", fake.synthesize_calls == 1, f"{fake.synthesize_calls}x")
    check(
        "user still gets an answer",
        result.get("final_response") == "好的，这是回答。",
        repr(result.get("final_response")),
    )
    degraded_context = human_text(fake.synthesize_messages[-1]) if fake.synthesize_messages else ""
    check(
        "degraded fields absent from context",
        all(m not in degraded_context for m in ("天气信息", "用户数据", "知识库检索结果")),
        degraded_context[:160],
    )

    print("\n[6] classification fallbacks")
    # Low confidence escalates to the complex fan-out.
    fake = install_stubs()
    app = build_workflow(ingest=False)
    run(app, fake, "模糊的问题", "product", "t-lowconf", confidence=0.2)
    low_conf_context = human_text(fake.synthesize_messages[-1])
    check(
        "low confidence escalates product -> complex fan-out",
        all(m in low_conf_context for m in ("天气信息", "用户数据", "知识库检索结果")),
        low_conf_context[:160],
    )

    # High confidence stays on the single path.
    fake = install_stubs()
    app = build_workflow(ingest=False)
    run(app, fake, "明确的问题", "product", "t-hiconf", confidence=0.95)
    check(
        "high confidence keeps the single path",
        "天气信息" not in human_text(fake.synthesize_messages[-1]),
    )

    # A classifier that raises must not take the graph down.
    class _ExplodingStructured:
        def invoke(self, prompt):
            raise RuntimeError("rate limited")

    fake = install_stubs()
    fake.with_structured_output = lambda schema: _ExplodingStructured()
    captured = {}

    class _RecordingRag2:
        def invoke(self, args):
            captured["query"] = args["query"]
            return "检索结果"

    nodes.rag_summarize = _RecordingRag2()
    app = build_workflow(ingest=False)
    try:
        result = run(app, fake, "分类会炸的问题", "product", "t-clsfail")
        crashed = False
    except Exception as e:  # noqa: BLE001
        result, crashed = {}, repr(e)
    check("classifier exception does not crash the graph", crashed is False, str(crashed))
    check(
        "classifier exception degrades to rag with the raw query",
        captured.get("query") == "分类会炸的问题",
        repr(captured.get("query")),
    )

    print("\n[7] identity and month come from the session and the query")
    recorded = {}

    class _RecordingFetch:
        def invoke(self, args):
            recorded.update(args)
            return "覆盖率:85%"

    fake = install_stubs()
    nodes.fetch_external_data = _RecordingFetch()
    app = build_workflow(ingest=False)
    run(app, fake, "我6月的报告", "report", "t-id", month="2025-06", user_id="1003")
    check("uses the selected user", recorded.get("user_id") == "1003", repr(recorded))
    check("uses the extracted month", recorded.get("month") == "2025-06", repr(recorded))

    # No month named -> latest month present in the data.
    recorded.clear()
    fake = install_stubs()
    nodes.fetch_external_data = _RecordingFetch()
    app = build_workflow(ingest=False)
    run(app, fake, "我的报告", "report", "t-nomonth", month="", user_id="1005")
    check(
        "empty month falls back to the latest in data",
        recorded.get("month") == available_months()[-1],
        f"{recorded.get('month')} vs {available_months()[-1]}",
    )

    # Unknown user -> falls back rather than raising.
    recorded.clear()
    fake = install_stubs()
    nodes.fetch_external_data = _RecordingFetch()
    app = build_workflow(ingest=False)
    run(app, fake, "我的报告", "report", "t-baduser", user_id="9999")
    check(
        "unknown user falls back to a real one",
        recorded.get("user_id") in available_user_ids(),
        repr(recorded.get("user_id")),
    )

    print("\n[8] history never carries empty text blocks into the next turn")

    def empty_text_blocks(msg):
        """Blocks Anthropic rejects with 'text content blocks must be non-empty'."""
        content = msg.content
        if isinstance(content, list):
            return [
                b
                for b in content
                if isinstance(b, dict)
                and b.get("type") == "text"
                and not str(b.get("text", "")).strip()
            ]
        return [] if str(content).strip() else [content]

    class _BlockStyleLLM(FakeLLM):
        """Streams the way langchain-anthropic does against Opus 5: indexed content
        blocks, where the block at index 0 stays empty (thinking is on by default and
        display='omitted') and the answer lands at index 1.

        Accumulation merges by index, so the empty block SURVIVES into the stored
        message — `.text` looks fine while the raw content still carries it. That is
        precisely what produced the observed
        400 'messages: text content blocks must be non-empty' on the following turn.
        """

        def stream(self, messages):
            self.synthesize_calls += 1
            self.synthesize_messages.append(messages)
            yield AIMessageChunk(content=[{"type": "text", "text": "", "index": 0}])
            yield AIMessageChunk(
                content=[{"type": "text", "text": "建议每周清洁。", "index": 1}]
            )

    fake = _BlockStyleLLM()
    nodes.classification_model = fake
    nodes.synthesis_model = fake
    nodes.rag_summarize = _FakeTool("滚刷建议每周清洁一次。")
    app = build_workflow(ingest=False)
    run(app, fake, "滚刷多久清洁一次？", "product", "t-blocks")
    stored = app.get_state({"configurable": {"thread_id": "t-blocks"}}).values["messages"]
    check(
        "stored assistant message has no empty text block",
        all(not empty_text_blocks(m) for m in stored),
        repr([m.content for m in stored]),
    )
    check(
        "stored assistant message keeps the text",
        any(m.text == "建议每周清洁。" for m in stored),
        repr([m.text for m in stored]),
    )
    # The real failure mode: turn 2 replays turn 1's history.
    run(app, fake, "那滤网呢？", "product", "t-blocks", standalone="滤网多久清洁一次？")
    replayed = fake.synthesize_messages[-1]
    check(
        "turn 2 replays no empty text block",
        all(not empty_text_blocks(m) for m in replayed),
        repr([m.content for m in replayed]),
    )

    # A turn that produces no text at all must not be recorded.
    class _SilentLLM(FakeLLM):
        def stream(self, messages):
            self.synthesize_calls += 1
            self.synthesize_messages.append(messages)
            yield AIMessageChunk(content=[{"type": "text", "text": "", "index": 0}])

    fake = _SilentLLM()
    nodes.classification_model = fake
    nodes.synthesis_model = fake
    nodes.rag_summarize = _FakeTool("检索结果")
    app = build_workflow(ingest=False)
    run(app, fake, "空回答的问题", "product", "t-silent")
    stored = app.get_state({"configurable": {"thread_id": "t-silent"}}).values.get(
        "messages", []
    )
    check("an empty-text turn is not stored at all", stored == [], repr(stored))

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
