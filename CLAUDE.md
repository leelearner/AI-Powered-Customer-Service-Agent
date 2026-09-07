# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt

# Run the app (Streamlit UI)
streamlit run app.py

# Graph topology regression tests — stubs the LLM and tools, no API keys, no network
python tests/test_topology.py

# Exercise a single layer directly — each module has a __main__ block
python -m rag.rag_service      # end-to-end RAG query
python -m rag.vector_store     # ingest data/ then run a raw retriever query
python -m utils.prompt_loader  # dump all four prompt files
python -m utils.config_handler # verify YAML + ${ENV} resolution
```

[tests/test_topology.py](tests/test_topology.py) is the only test and deliberately needs no credentials: it swaps `agent.nodes.llm` and the tool objects for fakes, then drives the real compiled graph. Extend it when you touch routing, state, or history. There is no linter or build step; pytest is not installed, so it runs as a plain script.

**Always run from the repository root.** `config/chroma.yml`'s `persist_directory` is fed to `Chroma()` verbatim (unlike every other config path, which goes through `get_abs_path`), so running from a subdirectory silently creates a second, empty vector store.

Required env vars (loaded from `.env` by `utils/config_handler.py`): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENWEATHER_API_KEY`.

## Architecture

A LangGraph `StateGraph` over a robot-vacuum knowledge base. See [graph_architecture.md](graph_architecture.md) for the design rationale (in Chinese).

### Routing lives in nodes, not in the graph

[agent/workflow.py](agent/workflow.py) declares only two edges: `START → intent_router` and `synthesize → END`. Every other transition is a runtime `Command(goto=...)` returned from a node function in [agent/nodes.py](agent/nodes.py). Reading `workflow.py` alone tells you nothing about execution order.

`intent_router` classifies the query into `weather` / `report` / `product` / `complex` via `llm.with_structured_output(QueryClassification)` and returns a `goto` that is either one node or a **list** of nodes, which LangGraph fans out in parallel:

- `weather` → `get_location` → `get_weather` → `synthesize`
- `report` → `[get_user_id, get_month]` → `fetch_data` → `synthesize`
- `product` → `rag` → `synthesize`
- `complex` → `[get_location, rag, get_user_id, get_month]` → … → `synthesize`

Adding a node means three edits: a function in `nodes.py` returning `Command(goto=...)`, a `graph.add_node(...)` in `workflow.py`, and a `goto` target from whichever upstream node should reach it.

Except for `messages`, `GraphState` ([agent/state.py](agent/state.py)) has no reducer annotations, so parallel branches must write **disjoint** keys — concurrent writes to the same key will raise.

### `synthesize` is deferred, and that is load-bearing

`synthesize` is registered with `defer=True`. Branches reach it at different depths (`rag` at step 2, `get_weather`/`fetch_data` at step 3), and without the barrier the `complex` path runs `synthesize` **twice** — once on partial context, then again on the full context, burning an extra LLM call. `tests/test_topology.py` asserts one execution per intent; do not drop the flag.

By contrast, `get_user_id` and `get_month` both `goto="fetch_data"` and need no barrier: they sit in the same superstep, so Pregel semantics already join them.

### Multi-turn state

`build_workflow()` compiles with an `InMemorySaver` by default, so history lives as long as the compiled graph object (`app.py` caches it in `st.session_state`). Every call must pass `config={"configurable": {"thread_id": ...}}` or turns from different users collide.

Two consequences worth internalising:

- **State persists across turns, so `intent_router` explicitly resets** `weather` / `external_data` / `rag_result` / `location` / `user_id` / `month` / `is_report` on every turn. Without that reset, a weather question followed by a product question injects the stale weather into the second answer. There is a regression test for exactly this.
- **`messages` uses the `add_messages` reducer**, so `synthesize_node` returns only the current turn's delta, never the accumulated list. It stores the *raw* user query rather than the context-augmented prompt, keeping history strictly user/assistant alternating.
- **Store plain text in history, never the accumulated `AIMessageChunk`.** The chunk keeps Anthropic's indexed block structure, and Opus 5 runs adaptive thinking by default with `display: "omitted"`, so a block can arrive with empty text and survive accumulation (merging is by index — `.text` reads fine while the raw content still carries the empty block). Replaying it fails the *next* turn with `400 messages: text content blocks must be non-empty`, and with an in-memory checkpointer that poisons every remaining turn of the session. `synthesize_node` therefore stores `AIMessage(content=final_response)` and skips the turn entirely when the text is blank — note `if response` does not catch that, since a message object is always truthy. `_recent_history` also filters blank messages so an already-poisoned thread recovers. Regression test: group [8].

`intent_router` also folds coreference resolution into its existing structured-output call, emitting `standalone_query` alongside `intent` — so multi-turn support costs no extra LLM round trip. `rag_node` retrieves with `standalone_query`, not the raw query; a follow-up like "那滤网呢？" has nothing retrievable on its own.

### Failure handling: degrade, never crash

The invariant is **the user always gets an answer**. Nodes that touch anything external are wrapped in `@_degrade_on_error(goto=..., **fallback)`, which logs, writes empty values, and continues to the declared next node instead of raising. `synthesize_node` builds its background section only from non-empty fields, so a degraded node simply drops out of the context. `intent_router` catches a failing classification call and falls back to the `rag` path; `synthesize_node` catches its own LLM failure and leaves `final_response` empty, which `app.py` renders as a retry message. `tests/test_topology.py` group [5] asserts the whole graph completes with *every* external call raising.

**Retry lives in the HTTP session, not on the node.** LangGraph's node-level `retry_policy` wraps a node from the outside, but `_degrade_on_error` catches inside it — so a retry policy on these nodes would never fire. Transient-failure retry is therefore configured on `http_session` in `agent_tools.py`. Don't add `retry_policy` to a degraded node expecting it to work.

Low classifier `confidence` (below `classification_confidence_threshold` in `config/agent.yml`) escalates routing to the `complex` fan-out rather than committing to one path — an extra call or two is cheaper than answering the wrong question.

### Streaming

`synthesize_node` accumulates `llm.stream(...)` rather than calling `invoke`, because LangGraph's `stream_mode="messages"` only emits tokens as they arrive if the underlying call actually streams. [app.py](app.py) consumes that stream and filters on **both** `metadata["langgraph_node"] == "synthesize"` (to suppress classifier and RAG-summariser tokens) and `isinstance(chunk, AIMessageChunk)` (to suppress the `HumanMessage` the node returns in its state update). Use `.text` on messages, not `.content` — Anthropic returns typed blocks, and `.text` concatenates just the text ones.

### `synthesize` absorbed the old middleware

The former ReAct middleware (`log_before_model`, `report_prompt_switch`) is now inlined in `synthesize_node`: it logs, picks `load_report_prompt()` vs `load_system_prompt()` off `state["is_report"]`, string-concatenates whichever of `weather` / `external_data` / `rag_result` are populated into a `背景信息:` block, prepends the trimmed conversation history, and streams the LLM response. Node-level tracing is just `logger.info` calls inside each node.

### One model factory, three roles

[chat/model/factory.py](chat/model/factory.py) is the only place a chat client is constructed. `ChatModelFactory(role)` reads `config/models.yml` and exports `synthesis_model`, `classification_model`, `rag_model`, `embedding_model` as module-level singletons. Change a model by editing that YAML — never by constructing a client inline.

The split is deliberate: `classification_model` runs the short structured output on every turn and is pointed at a cheap fast model at `temperature: 0` (a discriminative task), while `synthesis_model` produces what the user actually reads. Model IDs never carry a date suffix.

**`temperature` is optional and model-gated.** The current Opus/Sonnet generation rejects sampling params with a 400; Haiku still accepts them. The factory sends `temperature` only when the role's config declares it, so Opus/Sonnet roles simply omit the key. Adding it back to an Opus role breaks every call from that model.

### Tools vs. nodes

[agent/tools/agent_tools.py](agent/tools/agent_tools.py) holds the business logic as `@tool`-decorated functions. Nodes call them with `.invoke({...})`, never as plain functions. The `@tool` wrapper is now vestigial for the graph (no LLM binds these tools) but is kept so the logic stays independently testable.

`get_user_id_tool`, `get_random_user_id` and `get_current_month` are **deprecated and unused** — they returned random values, so "my June report" could return another user's March data. Identity now flows in as `session_user_id` (a sidebar selection) and the month comes from the classifier's `month` field, both validated against `available_user_ids()` / `available_months()`, which read the actual CSV.

Outbound HTTP goes through `http_session`, a `requests.Session` with a retry adapter, and every call passes `timeout=HTTP_TIMEOUT`.

`agent/react_agent.py` and `agent/tools/middleware.py` are fully commented out; they are the pre-graph implementation kept for reference.

### RAG ingestion and the MD5 ledger

[rag/vector_store.py](rag/vector_store.py) walks `data/` for `.txt`/`.pdf`, and skips any file whose MD5 already appears in `md5.text`. **If you delete `chroma_db/`, delete `md5.text` too** — otherwise ingestion no-ops and every retrieval comes back empty.

Ingestion is a separate `ingest()` method, not a constructor side effect, and `get_rag_service()` is an `lru_cache` singleton. `build_workflow()` calls `ingest()` once at build time; pass `ingest=False` in tests. Constructing the service must stay free of disk scanning — it used to re-hash every file in `data/` on every RAG query.

### Configuration

[utils/config_handler.py](utils/config_handler.py) loads all four YAML files at import time into module-level dicts (`models_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`) and expands `${VAR}` against the environment. A new config file needs a `load_*_config()` function plus its module-level singleton. Paths inside YAML are repo-root-relative and must be passed through `get_abs_path()` at the use site.

Prompts are plain `.txt` under `prompts/`, registered in `config/prompts.yml`, and read by a dedicated `load_*_prompt()` in [utils/prompt_loader.py](utils/prompt_loader.py). `rag_summarize.txt` uses `{input}`/`{context}` and `classification_prompt.txt` uses `{input}`/`{history}` as `str.format`/`PromptTemplate` placeholders — literal braces in those files will break formatting.

`classification_prompt.txt` now specifies a four-field structured object (`intent`, `standalone_query`, `topic`, `summary`) matching `QueryClassification`. It previously said "仅输出一个单词", which contradicted `with_structured_output` — keep the prompt's output contract and the TypedDict in sync when editing either.

## Conventions

- Prompts, knowledge documents, and user-facing strings are Chinese; code, comments, and log messages are English.
- Log via `from utils.logger_handler import logger` — a shared singleton writing to console and `logs/agent_YYYYMMDD.log`.
- `README.md` refers to `model/factory.py`; the actual path is `chat/model/factory.py`.
