[English](README.md) | [中文](README_zh.md)
# RAG + Agent Intelligent Q&A System

An intelligent Q&A system built with LangChain Agent + RAG. Using a robot vacuum knowledge base as an example scenario, it demonstrates how to combine Retrieval-Augmented Generation (RAG) with a ReAct Agent to build a conversational assistant with tool calling, context awareness, and dynamic prompt switching capabilities.

---

## Technical Highlights

### RAG Offline Document Ingestion

- Supports `.txt` and `.pdf` knowledge documents stored in the `data/` directory
- Uses `RecursiveCharacterTextSplitter` for document chunking (chunk_size / chunk_overlap configurable)
- **MD5 incremental checksum** mechanism (`md5.text`) prevents duplicate ingestion, enabling idempotent offline updates
- Vectorization uses OpenAI `text-embedding-3-small`, with persistent storage in local **ChromaDB**

### RAG Online Retrieval

- Each query retrieves Top-K relevant document chunks via ChromaDB retriever (K is configurable)
- Retrieved results are concatenated into structured context, injected into the RAG Prompt Template, and the LLM generates the final answer
- RAG capability is encapsulated as a LangChain `@tool` for on-demand invocation by the Agent

### LangChain Tool Definitions

The Agent is equipped with the following tools:

| Tool | Description |
|------|-------------|
| `rag_summarize` | Retrieves from the knowledge base via RAG and generates a summarized answer |
| `get_weather` | Fetches weather information for a specified city |
| `get_user_location` | Gets the user's current city |
| `get_user_id` | Gets the current user ID |
| `get_current_month` | Gets the current month |
| `fetch_external_data` | Reads external business data (CSV) based on user ID and month |
| `fill_context_for_report` | Triggers the report mode context flag |

### Middleware

Cross-cutting logic is injected into the Agent's execution pipeline via the LangChain Agent Middleware mechanism:

- **`monitor_tool`** (`@wrap_tool_call`): Intercepts all tool calls, logging tool names and input parameters; when `fill_context_for_report` is invoked, it writes a report flag to the runtime context
- **`log_before_model`** (`@before_model`): Logs the current message count and the latest message content before each LLM call
- **`report_prompt_switch`** (`@dynamic_prompt`): Dynamically switches the system prompt based on the `report` flag in the runtime context, enabling runtime switching between normal conversation and report generation modes

### RAG and Agent Integration

`rag_summarize` is registered as a regular tool for the Agent, which autonomously decides whether to query the knowledge base during reasoning. This design allows RAG and other tools (weather, user info, external data, etc.) to work together within a single conversation. For example:

> "How should I maintain my robot vacuum given the temperature in my area?"
> → The Agent first calls `get_user_location` to get the city → calls `get_weather` to get the temperature → calls `rag_summarize` to retrieve maintenance knowledge → synthesizes and generates a comprehensive answer

### Miscellaneous

- **Models**: LLM uses Anthropic Claude (`claude-sonnet-4-6`), Embedding uses OpenAI (`text-embedding-3-small`), both managed via a unified factory class
- **Configuration**: All parameters are managed through YAML files (`config/`), with zero hardcoding in business logic
- **Logging**: Unified logging module with date-based rolling file output to `logs/`
- **Frontend**: Streamlit provides the interactive UI

---

## Project Structure

```
.
├── app.py                  # Streamlit entry point
├── agent/
│   ├── react_agent.py      # ReAct Agent definition
│   └── tools/
│       ├── agent_tools.py  # Tool definitions
│       └── middleware.py   # Middleware definitions
├── rag/
│   ├── rag_service.py      # RAG retrieval and generation chain
│   └── vector_store.py     # ChromaDB vector store and document ingestion
├── model/
│   └── factory.py          # LLM / Embedding factory
├── utils/                  # Config, logging, file, and path utilities
├── config/                 # YAML configuration files
├── prompts/                # Prompt templates
└── data/                   # Knowledge documents
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
export OPENAI_API_KEY=your_openai_api_key
```

### 3. Add Knowledge Documents

Place `.txt` or `.pdf` documents in the `data/` directory. Vectorization and ingestion happen automatically on the first run; newly added files are incrementally processed on subsequent startups.

### 4. Run the Application

**Web Interface (Streamlit):**

```bash
streamlit run app.py
```

**Run the Agent directly from the command line:**

```bash
python -m agent.react_agent
```

**Test RAG retrieval only:**

```bash
python -m rag.rag_service
```

### 5. Configuration

All module parameters can be modified in the YAML files under the `config/` directory:

| File | Description |
|------|-------------|
| `config/rag.yml` | LLM and Embedding model names |
| `config/chroma.yml` | Vector store path, chunking parameters, retrieval Top-K, etc. |
| `config/prompts.yml` | Prompt template file paths |
| `config/agent.yml` | External data file paths |
