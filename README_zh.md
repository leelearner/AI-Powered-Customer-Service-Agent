[English](README.md) | [中文](README_zh.md)
# RAG + Agent 智能问答系统

基于 LangChain Agent + LangGraph + RAG 构建的智能问答系统，以扫地机器人知识库为示例场景，演示如何将检索增强生成（RAG）与 Graph 化的 Agent 工作流结合，实现具备意图路由、工具调用、上下文感知和动态提示词切换能力的对话助手。

---

## 架构概览

系统采用 LangGraph StateGraph 架构，通过显式的图节点和边替代原 ReAct Agent 的隐式工具调用链，使执行路径可控、可审计。

![Workflow Graph](./graph.png)

详细架构设计参见 [graph_architecture.md](graph_architecture.md)。

---

## 技术要点

### LangGraph Workflow

- 使用 `StateGraph` 定义节点和边，`intent_router` 通过 LLM 结构化输出分类用户意图
- 四种意图路径：天气查询（`weather`）、使用报告（`report`）、产品知识（`product`）、复合查询（`complex`）
- 通过 `Command(goto=...)` 实现动态路由，支持并行节点调度
- 所有路径最终汇聚到 `synthesize` 节点生成回答

### RAG 离线文档入库

- 支持 `.txt` 和 `.pdf` 格式的知识文档，存放于 `data/` 目录
- 使用 `RecursiveCharacterTextSplitter` 对文档进行分块（chunk_size / chunk_overlap 可配置）
- 通过 **MD5 增量校验**机制（`md5.text`）避免重复入库，实现幂等的离线更新
- 向量化使用 OpenAI `text-embedding-3-small`，持久化存储至本地 **ChromaDB**

### RAG 在线检索

- 每次查询通过 ChromaDB retriever 检索 Top-K 相关文档片段（K 可配置）
- 检索结果拼接为结构化 context，注入 RAG Prompt Template，由 LLM 生成最终答案
- RAG 能力以 LangChain `@tool` 形式封装，作为图节点 `rag` 按需调用

### Graph 节点

| 节点 | 说明 |
|------|------|
| `intent_router` | LLM 意图分类，动态路由到对应节点 |
| `get_location` | 根据 IP 获取用户城市/经纬度 |
| `get_weather` | 根据经纬度获取天气信息 |
| `get_user_id` | 获取当前用户 ID |
| `get_month` | 获取当前月份 |
| `fetch_data` | 读取外部业务数据（CSV），触发报告模式 |
| `rag` | 基于 RAG 检索知识库并生成摘要 |
| `synthesize` | 汇总所有上下文，根据意图切换 prompt，生成最终回答 |

### RAG 与 Agent 结合

`rag` 作为 Graph 中的一个节点，由 `intent_router` 按意图分类结果决定是否调用。这种设计使得 RAG 与其他节点（天气、用户信息、外部数据等）可以并行工作，例如：

> "扫地机器人在我所在地区的气温下如何保养"
> → `intent_router` 判定为 `complex` → 并行执行 `get_location`、`rag`、`get_user_id`、`get_month` → 各路径完成后汇聚到 `synthesize` → 综合生成回答

### 其他

- **模型**：LLM 使用 Anthropic Claude（`claude-sonnet-4-6`），Embedding 使用 OpenAI（`text-embedding-3-small`）
- **配置**：所有参数通过 YAML 文件管理（`config/`），业务代码零硬编码
- **日志**：统一日志模块，按日期滚动写入 `logs/`
- **前端**：使用 Streamlit 提供交互界面

---

## 项目结构

```
.
├── app.py                  # Streamlit 入口，调用 workflow 执行图
├── graph.png               # LangGraph 自动生成的工作流图
├── graph_architecture.md   # 架构设计文档
├── agent/
│   ├── state.py            # GraphState / QueryClassification 定义
│   ├── nodes.py            # 图节点函数（意图路由、工具调用、回答合成）
│   ├── workflow.py         # StateGraph 组装与编译
│   ├── react_agent.py      # 原 ReAct Agent（已弃用）
│   └── tools/
│       ├── agent_tools.py  # 业务工具（@tool 装饰）
│       └── middleware.py   # 原 Middleware（已弃用，逻辑内联到 nodes.py）
├── rag/
│   ├── rag_service.py      # RAG 检索与生成链
│   └── vector_store.py     # ChromaDB 向量存储与文档入库
├── model/
│   └── factory.py          # LLM / Embedding 工厂
├── utils/                  # 配置、日志、文件、路径等工具
├── config/                 # YAML 配置文件
├── prompts/                # Prompt 模板
└── data/                   # 知识文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export ANTHROPIC_API_KEY=your_anthropic_api_key
export OPENAI_API_KEY=your_openai_api_key
```

### 3. 添加知识文档

将 `.txt` 或 `.pdf` 文档放入 `data/` 目录。首次运行时会自动完成向量化入库；后续新增文件也会在启动时增量处理。

### 4. 启动应用

**Web 界面（Streamlit）：**

```bash
streamlit run app.py
```

**仅测试 RAG 检索：**

```bash
python -m rag.rag_service
```

### 5. 配置调整

各模块参数均可在 `config/` 目录下的 YAML 文件中修改：

| 文件 | 说明 |
|------|------|
| `config/rag.yml` | LLM 和 Embedding 模型名称 |
| `config/chroma.yml` | 向量库路径、分块参数、检索 Top-K 等 |
| `config/prompts.yml` | Prompt 模板文件路径 |
| `config/agent.yml` | 外部数据文件路径 |
