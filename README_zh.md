[English](README.md) | [中文](README_zh.md)
# RAG + Agent 智能问答系统

基于 LangChain Agent + RAG 构建的智能问答系统，以扫地机器人知识库为示例场景，演示如何将检索增强生成（RAG）与 ReAct Agent 结合，实现具备工具调用、上下文感知和动态提示词切换能力的对话助手。

---

## 技术要点

### RAG 离线文档入库

- 支持 `.txt` 和 `.pdf` 格式的知识文档，存放于 `data/` 目录
- 使用 `RecursiveCharacterTextSplitter` 对文档进行分块（chunk_size / chunk_overlap 可配置）
- 通过 **MD5 增量校验**机制（`md5.text`）避免重复入库，实现幂等的离线更新
- 向量化使用 OpenAI `text-embedding-3-small`，持久化存储至本地 **ChromaDB**

### RAG 在线检索

- 每次查询通过 ChromaDB retriever 检索 Top-K 相关文档片段（K 可配置）
- 检索结果拼接为结构化 context，注入 RAG Prompt Template，由 LLM 生成最终答案
- RAG 能力以 LangChain `@tool` 形式封装，供 Agent 按需调用

### LangChain Tool 封装

Agent 配备以下工具：

| 工具 | 说明 |
|------|------|
| `rag_summarize` | 基于 RAG 检索知识库并生成摘要回答 |
| `get_weather` | 获取指定城市的天气信息 |
| `get_user_location` | 获取用户所在城市 |
| `get_user_id` | 获取当前用户 ID |
| `get_current_month` | 获取当前月份 |
| `fetch_external_data` | 根据用户 ID 和月份读取外部业务数据（CSV） |
| `fill_context_for_report` | 触发报告模式上下文标记 |

### Middleware 中间件

通过 LangChain Agent Middleware 机制在 Agent 运行链路中注入横切逻辑：

- **`monitor_tool`**（`@wrap_tool_call`）：拦截所有工具调用，记录工具名称与入参；当 `fill_context_for_report` 被调用时，向运行时 context 写入报告标记
- **`log_before_model`**（`@before_model`）：在每次 LLM 调用前记录当前消息数量及最新消息内容
- **`report_prompt_switch`**（`@dynamic_prompt`）：根据运行时 context 中的 `report` 标志动态切换系统提示词，实现普通对话与报告生成两种模式的运行时切换

### RAG 与 Agent 结合

`rag_summarize` 作为 Agent 的一个普通工具注册，Agent 在推理过程中自主判断是否需要检索知识库。这种设计使得 RAG 与其他工具（天气、用户信息、外部数据等）可以在同一次对话中协同工作，例如：

> "扫地机器人在我所在地区的气温下如何保养"
> → Agent 先调用 `get_user_location` 获取城市 → 调用 `get_weather` 获取气温 → 调用 `rag_summarize` 检索保养知识 → 综合生成回答

### 其他

- **模型**：LLM 使用 Anthropic Claude（`claude-sonnet-4-6`），Embedding 使用 OpenAI（`text-embedding-3-small`），均通过工厂类统一管理
- **配置**：所有参数通过 YAML 文件管理（`config/`），业务代码零硬编码
- **日志**：统一日志模块，按日期滚动写入 `logs/`
- **前端**：使用 Streamlit 提供交互界面

---

## 项目结构

```
.
├── app.py                  # Streamlit 入口
├── agent/
│   ├── react_agent.py      # ReAct Agent 定义
│   └── tools/
│       ├── agent_tools.py  # 工具定义
│       └── middleware.py   # Middleware 定义
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

**命令行直接运行 Agent：**

```bash
python -m agent.react_agent
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
