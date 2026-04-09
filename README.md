[English](README.md) | [中文](README_zh.md)
# RAG + Agent Intelligent Q&A System

An intelligent Q&A system built with LangChain Agent + LangGraph + RAG. Using a robot vacuum knowledge base as an example scenario, it demonstrates how to combine Retrieval-Augmented Generation (RAG) with a Graph-based Agent workflow to build a conversational assistant with intent routing, tool calling, context awareness, and dynamic prompt switching capabilities.

---

## Architecture Overview

The system uses a LangGraph StateGraph architecture, replacing the implicit tool calling chain of the original ReAct Agent with explicit graph nodes and edges, making execution paths controllable and auditable.

![Workflow Graph](./graph.png)

For detailed architecture design, see [graph_architecture.md](graph_architecture.md).

---

## Technical Highlights

### LangGraph Workflow

- Uses `StateGraph` to define nodes and edges; `intent_router` classifies user intent via LLM structured output
- Four intent paths: weather query (`weather`), usage report (`report`), product knowledge (`product`), complex query (`complex`)
- Dynamic routing via `Command(goto=...)`, supporting parallel node scheduling
- All paths converge at the `synthesize` node for final response generation

### RAG Offline Document Ingestion

- Supports `.txt` and `.pdf` knowledge documents stored in the `data/` directory
- Uses `RecursiveCharacterTextSplitter` for document chunking (chunk_size / chunk_overlap configurable)
- **MD5 incremental checksum** mechanism (`md5.text`) prevents duplicate ingestion, enabling idempotent offline updates
- Vectorization uses OpenAI `text-embedding-3-small`, with persistent storage in local **ChromaDB**

### RAG Online Retrieval

- Each query retrieves Top-K relevant document chunks via ChromaDB retriever (K is configurable)
- Retrieved results are concatenated into structured context, injected into the RAG Prompt Template, and the LLM generates the final answer
- RAG capability is encapsulated as a LangChain `@tool`, invoked as the `rag` graph node on demand

### Graph Nodes

| Node | Description |
|------|-------------|
| `intent_router` | LLM intent classification, dynamically routes to target nodes |
| `get_location` | Gets user city/coordinates based on IP |
| `get_weather` | Fetches weather information based on coordinates |
| `get_user_id` | Gets the current user ID |
| `get_month` | Gets the current month |
| `fetch_data` | Reads external business data (CSV), triggers report mode |
| `rag` | Retrieves from knowledge base via RAG and generates summary |
| `synthesize` | Aggregates all context, switches prompt based on intent, generates final response |

### RAG and Agent Integration

`rag` is a node in the Graph, invoked by `intent_router` based on intent classification. This design allows RAG and other nodes (weather, user info, external data, etc.) to work in parallel. For example:

> "How should I maintain my robot vacuum given the temperature in my area?"
> → `intent_router` classifies as `complex` → parallel execution of `get_location`, `rag`, `get_user_id`, `get_month` → all paths converge at `synthesize` → generates comprehensive response

### Miscellaneous

- **Models**: LLM uses Anthropic Claude (`claude-sonnet-4-6`), Embedding uses OpenAI (`text-embedding-3-small`)
- **Configuration**: All parameters are managed through YAML files (`config/`), with zero hardcoding in business logic
- **Logging**: Unified logging module with date-based rolling file output to `logs/`
- **Frontend**: Streamlit provides the interactive UI

---

## Project Structure

```
.
├── app.py                  # Streamlit entry point, invokes workflow graph
├── graph.png               # Auto-generated workflow graph from LangGraph
├── graph_architecture.md   # Architecture design document
├── agent/
│   ├── state.py            # GraphState / QueryClassification definitions
│   ├── nodes.py            # Graph node functions (intent routing, tool calls, synthesis)
│   ├── workflow.py         # StateGraph assembly and compilation
│   ├── react_agent.py      # Original ReAct Agent (deprecated)
│   └── tools/
│       ├── agent_tools.py  # Business tools (@tool decorated)
│       └── middleware.py   # Original Middleware (deprecated, logic inlined into nodes.py)
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
