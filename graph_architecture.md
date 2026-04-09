# Graph 架构设计文档

## 一、背景：原 ReAct 架构的问题

原架构使用 `create_agent` 创建一个 ReAct Agent，所有工具调用顺序完全由 LLM 自主决定：

```
用户 Query
    ↓
ReactAgent (LLM 自主决策)
    ↓
[工具池]: rag_summarize / get_user_location / get_weather /
          get_user_id / get_current_month / fetch_external_data /
          fill_context_for_report
    ↓
Middleware 拦截 (monitor_tool / log_before_model / report_prompt_switch)
    ↓
流式输出
```

**核心问题**：工具间存在隐含依赖（如 `get_weather` 依赖 `get_user_location` 的输出），这些依赖靠 LLM 推理维护，不可控、不可审计。

---

## 二、工具间真实数据依赖

```
ip (来自 metadata)
  └─► get_user_location(ip) ──► city, lat, lon
                                    └─► get_weather(city, lat, lon)

get_user_id() ──────────────┐
                             ├─► fetch_external_data(user_id, month)
get_current_month() ─────────┘

rag_summarize(query) ──► RAG 知识库检索结果

fill_context_for_report() ──► 触发 context["report"]=True
                                  └─► report_prompt_switch 切换 system prompt
```

Graph 架构的目标：将上述隐含依赖**显式编码为图的边**。

---

## 三、Graph 节点设计

### State 定义

```python
class GraphState(TypedDict):
    query: str
    ip: str
    location: dict        # city, lat, lon
    weather: str
    user_id: str
    month: str
    external_data: str
    rag_result: str
    is_report: bool
    messages: list
    final_response: str
```

### 节点列表

| 节点 | 职责 | 对应原工具/中间件 |
|------|------|-----------------|
| `intent_router` | 分析 query 意图，决定路径 | LLM 判断（原来隐式） |
| `get_location_node` | 根据 IP 获取城市/经纬度 | `get_user_location` |
| `get_weather_node` | 根据经纬度获取天气 | `get_weather` |
| `get_user_id_node` | 获取用户 ID | `get_user_id` |
| `get_month_node` | 获取当前月份 | `get_current_month` |
| `fetch_data_node` | 获取外部用户数据 | `fetch_external_data` + `fill_context_for_report` |
| `rag_node` | RAG 检索知识库 | `rag_summarize` |
| `synthesize_node` | 汇总上下文，生成最终回答 | LLM + `log_before_model` + `report_prompt_switch` |

### Middleware 的归宿

| 原 Middleware | Graph 中的位置 |
|--------------|---------------|
| `monitor_tool` | 每个 node 函数内部的日志逻辑，或统一 node wrapper |
| `log_before_model` | `synthesize_node` 调用 LLM 前执行 |
| `report_prompt_switch` | `synthesize_node` 内读取 `state["is_report"]` 切换 prompt |

---

## 四、图结构与节点关系

```
START
  ↓
[intent_router]  ── 条件路由 ──►
  │
  ├── "weather"  ──► [get_location_node] ──► [get_weather_node]
  │                                               ↓
  ├── "report"   ──► [get_user_id_node] ──┐       │
  │                  [get_month_node]  ──┘ ├─► [fetch_data_node]
  │                                         │       ↓
  ├── "product"  ──► [rag_node] ───────────┼───────┤
  │                                         │       │
  └── "complex"  ──► [get_location_node]   │       │
                     [rag_node]        并行─┤       │
                     [get_user_id_node]    │       │
                         ↓                 │       │
                     [get_weather_node]    │       │
                     [fetch_data_node] ───┘       │
                                                   ↓
                                           [synthesize_node]
                                                   ↓
                                                  END
```

---

## 五、文件组织架构

### 目录结构

```
agent/
  state.py          # GraphState TypedDict 定义
  nodes.py          # 所有 node 函数（对 tools 的图层面薄包装）
  workflow.py       # StateGraph 组装，返回 CompiledStateGraph
  tools/
    agent_tools.py  # 纯业务逻辑（保留，可独立测试和复用）

app.py              # 调用入口，直接使用 workflow.py 暴露的 compiled graph
```

### 各文件职责

**`agent/state.py`**
- 定义 `GraphState`，作为所有节点间共享的数据契约

**`agent/tools/agent_tools.py`**
- 保留原有纯业务函数（HTTP 调用、RAG 检索等）
- 函数签名不变，接收业务参数，返回业务结果
- 可独立单元测试，不依赖图结构

**`agent/nodes.py`**
- 每个 node 函数接收 `GraphState`，调用 `tools/` 中的业务函数，返回 state 更新
- 包含原 middleware 逻辑（日志、prompt 切换）内联到对应节点
- Node 与 Tool 的分层示例：
  ```python
  # tool（在 agent_tools.py）
  def get_weather(city: str, lat: str, lon: str) -> str:
      ...

  # node（在 nodes.py）
  def get_weather_node(state: GraphState) -> dict:
      result = get_weather(
          state["location"]["city"],
          state["location"]["lat"],
          state["location"]["lon"],
      )
      return {"weather": result}
  ```

**`agent/workflow.py`**
- 唯一职责：声明节点、声明边、编译图
- 暴露 `build_workflow() -> CompiledStateGraph`
- 不包含任何业务逻辑

**`app.py`**
- 直接调用 `build_workflow()` 获取 compiled graph
- 处理输入构造和流式输出，不再需要 `ReactAgent` 类

---

## 六、ReactAgent 类的去留

原 `ReactAgent` 的三个职责在 Graph 架构下的归宿：

| 职责 | 原位置 | Graph 后 |
|------|--------|---------|
| 组装 agent | `__init__` | `workflow.py` 的 `build_workflow()` |
| 构造输入 | `execute_stream` | `app.py` 直接构造初始 `GraphState` |
| 处理流式输出 | `execute_stream` yield | `app.py` 直接处理 graph stream |

**结论**：项目只有 `app.py` 一个调用入口，`ReactAgent` 类可以完全去掉。`workflow.py` 直接暴露 compiled graph 供 `app.py` 使用。若未来出现多个调用入口（CLI、定时任务等），可引入轻量门面类 `AgentWorkflow`，只做输入/输出适配，不含业务逻辑。
