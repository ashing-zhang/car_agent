# AutoAgent：面向 CARLA 仿真环境的多模态个性化车载 Agent

> **文档类型**：生产级 AI Coding 技术方案 / SPEC  
> **目标**：指导 AI Coding Agent 从零实现一个可运行、可测试、可演示、可扩展的 Automotive Agent 项目  
> **项目定位**：求职作品集级，而非自动驾驶算法比赛项目  
> **版本**：v1.0  
> **日期**：2026-09-01

---

## 1. 项目目标

实现一个运行于 **CARLA 车辆仿真环境**之上的多模态、个性化车载 Agent。

系统不是自动驾驶控制器，而是位于“用户 ↔ 智能座舱/车辆能力”之间的 Agent 层：

```text
User
 │
 ├── Text / Voice
 └── Camera / Vehicle Context
        │
        ▼
┌─────────────────────────────┐
│      Automotive Agent       │
│                             │
│ Intent / Planning / Tool Use│
│ Context / Memory / Policy   │
└──────────────┬──────────────┘
               │
       MCP / Internal Tools
               │
 ┌─────────────┼──────────────┐
 ▼             ▼              ▼
Vehicle      Navigation      Media
Adapter      Adapter         Adapter
 │             │              │
 └─────────────┼──────────────┘
               ▼
             CARLA
```

### 核心能力

1. 自然语言理解
2. Vehicle State 感知
3. CARLA 环境感知
4. Tool Calling
5. 多步任务规划
6. 个性化长期记忆
7. 多模态上下文
8. MCP Tool Server
9. Agent Evaluation
10. Observability
11. 可替换 LLM / VLM
12. 后续可扩展 ROS 2 / Autoware

CARLA 的 ROS bridge 支持 CARLA 与 ROS/ROS2 双向通信，并可提供 Camera、LiDAR、GNSS、Radar、IMU、交通灯、碰撞、车道侵入等数据；因此本项目第一阶段可以直接使用 CARLA Python API，第二阶段再增加 ROS2。 

---

# 2. 非目标

AI Coding Agent 必须严格避免项目范围膨胀。

本项目 **不是**：

- 自动驾驶算法
- ADAS 控制器
- SLAM 项目
- 车辆动力学研究
- 自动驾驶 Perception Benchmark
- 训练新的基础模型

不要实现：

- 自研目标检测模型
- 自研路径规划算法
- 自研车辆控制器
- 自研语音识别模型
- 自研大语言模型

这些能力只作为 Agent 的外部能力或仿真环境使用。

---

# 3. 核心 Demo

最终 README 至少展示以下 6 个 Demo。

## Demo A：车辆状态查询

用户：

> 现在车速多少？还有多少电？

Agent：

```text
get_vehicle_status()
```

输出：

```text
当前车速 63 km/h，电量 72%。
```

---

## Demo B：车辆控制

用户：

> 有点冷。

Agent：

1. 查询当前温度
2. 查询用户温度偏好
3. 决策目标温度
4. 调用 `set_temperature`
5. 返回结果

例如：

```text
Current temperature = 21℃
User preference = 24℃
→ set_temperature(24)
```

---

## Demo C：个性化

用户历史：

```text
“我冬天喜欢车内保持 24℃。”
```

写入 Memory：

```json
{
  "subject": "user",
  "predicate": "prefers_temperature",
  "object": 24,
  "valid_from": "...",
  "confidence": 0.92
}
```

以后：

> 有点冷。

Agent 自动结合历史偏好，而不是只依赖当前对话。

---

## Demo D：复杂任务

用户：

> 我要去公司，顺便播放我平时上班喜欢听的音乐。

Agent：

```text
1. retrieve user preference
2. search navigation destination
3. start navigation
4. retrieve media preference
5. play music
```

要求 Agent 能展示：

```text
Plan
Tool Calls
Tool Results
Final Response
```

---

## Demo E：多模态场景

CARLA Camera：

```text
rain
heavy traffic
road construction
```

用户：

> 现在还适合走原来的路线吗？

Agent：

```text
Vision Context
+
Navigation Context
+
Vehicle State
+
Weather
+
Memory
        ↓
Decision
```

注意：本项目只要求**理解仿真场景并做建议**，不要让 LLM 直接承担安全关键驾驶控制。

---

## Demo F：MCP

将 Vehicle / Navigation / Media 等能力暴露为 MCP Tools：

```text
Automotive Agent
       │
       ▼
MCP Client
       │
       ▼
Automotive MCP Server
       │
       ├── vehicle
       ├── navigation
       ├── media
       └── user_context
```

MCP 2026-07-28 规范支持通过 `tools/list` 发现工具，并由模型根据上下文调用工具；工具应具有明确的名称、描述和输入 schema。

---

# 4. 总体技术架构

```text
                         ┌───────────────┐
                         │     User      │
                         └───────┬───────┘
                                 │
                    Text / Voice / Image
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  API / Session   │
                       └────────┬─────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │ Context Aggregator  │
                     └─────────┬──────────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
              ▼                ▼                 ▼
        Vehicle State      Environment       Memory
              │                │                 │
              └────────────────┼─────────────────┘
                               ▼
                     ┌────────────────────┐
                     │  LangGraph Agent   │
                     │                    │
                     │ Router / Planner   │
                     │ Tool Selection     │
                     │ Reflection         │
                     └─────────┬──────────┘
                               │
                         Tool / MCP
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
        Vehicle Tool      Navigation Tool     Media Tool
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
                             CARLA

Parallel:
CARLA Camera ──> VLM ──> Environment Context
CARLA State  ───────────> Vehicle Context

Observability:
Agent ──> Trace / Metrics / Logs / Evaluation
```

LangGraph 适合这里的原因是它提供有状态 Agent 编排、持久化、streaming、human-in-the-loop 和长流程执行能力；本项目使用它作为 Agent Runtime，而不是把所有业务逻辑塞进一个 prompt。

---

# 5. 技术栈

## 必选

| 模块 | 技术 |
|---|---|
| Language | Python 3.11+ |
| Agent Runtime | LangGraph |
| LLM | OpenAI-compatible API / Qwen / 本地模型 |
| VLM | 可选 Qwen-VL 等 |
| Simulator | CARLA |
| Backend | FastAPI |
| Protocol | MCP |
| Database | PostgreSQL |
| Cache | Redis |
| Vector Retrieval | pgvector |
| ORM | SQLAlchemy |
| Validation | Pydantic |
| Testing | pytest |
| Container | Docker Compose |
| Observability | OpenTelemetry + Prometheus/Grafana 或等价方案 |

## 第二阶段

```text
ROS 2
CARLA ROS Bridge
```

CARLA 官方 ROS bridge 支持 ROS 2，并提供同步模式、车辆控制以及多类传感器数据接口。

---

# 6. Repository Structure

必须生成：

```text
autoagent/
├── README.md
├── SPEC.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── routes_agent.py
│   │   ├── routes_vehicle.py
│   │   └── routes_health.py
│   │
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── planner.py
│   │   ├── router.py
│   │   ├── policies.py
│   │   └── prompts/
│   │       ├── system.md
│   │       ├── planning.md
│   │       └── memory.md
│   │
│   ├── tools/
│   │   ├── vehicle.py
│   │   ├── navigation.py
│   │   ├── media.py
│   │   └── environment.py
│   │
│   ├── mcp/
│   │   ├── server.py
│   │   └── schemas.py
│   │
│   ├── carla/
│   │   ├── client.py
│   │   ├── vehicle_adapter.py
│   │   ├── sensor_manager.py
│   │   └── world_manager.py
│   │
│   ├── memory/
│   │   ├── models.py
│   │   ├── repository.py
│   │   ├── extractor.py
│   │   ├── retriever.py
│   │   ├── temporal.py
│   │   └── policy.py
│   │
│   ├── multimodal/
│   │   ├── asr.py
│   │   ├── vlm.py
│   │   └── context.py
│   │
│   ├── context/
│   │   ├── aggregator.py
│   │   └── schemas.py
│   │
│   ├── evaluation/
│   │   ├── dataset.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── scenarios/
│   │
│   └── observability/
│       ├── tracing.py
│       ├── metrics.py
│       └── logging.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── scripts/
│   ├── start_carla.sh
│   ├── seed_memory.py
│   └── run_eval.py
│
├── configs/
│   ├── dev.yaml
│   ├── eval.yaml
│   └── carla.yaml
│
└── docs/
    ├── architecture.md
    ├── memory.md
    ├── tools.md
    ├── evaluation.md
    └── deployment.md
```

---

# 7. Agent State

LangGraph State 必须显式定义。

```python
class AgentState(TypedDict):
    session_id: str
    user_id: str

    user_message: str

    vehicle_state: VehicleState | None
    environment_state: EnvironmentState | None
    navigation_state: NavigationState | None

    memories: list[Memory]
    plan: list[PlanStep]

    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResult]

    response: str

    error: str | None
```

禁止依赖隐式全局变量。

---

# 8. Vehicle State

统一 schema：

```python
class VehicleState(BaseModel):
    speed_kmh: float
    battery_percent: float
    cabin_temperature_c: float

    ac_enabled: bool
    target_temperature_c: float

    latitude: float
    longitude: float

    current_road: str | None

    timestamp: datetime
```

CARLA Adapter 必须负责把 CARLA 原生状态转换成该 schema。

Agent 不允许直接调用 CARLA API。

必须：

```text
Agent
 ↓
Tool Interface
 ↓
VehicleService
 ↓
CARLA Adapter
 ↓
CARLA
```

这样未来替换成真实车辆 API 时，Agent 层不需要修改。

---

# 9. Tool Design

至少实现以下工具：

## vehicle

```text
get_vehicle_status
set_temperature
set_ac
get_cabin_temperature
```

## navigation

```text
get_navigation_status
search_destination
start_navigation
cancel_navigation
```

## media

```text
play_media
pause_media
set_volume
```

## environment

```text
get_weather
get_traffic
get_camera_scene
```

---

# 10. Tool 安全策略

Agent **绝不能直接拥有任意 CARLA API 权限**。

所有工具必须经过：

```text
LLM
 ↓
Tool Schema Validation
 ↓
Policy Validation
 ↓
Execution
 ↓
Audit Log
```

例如：

```python
set_temperature(
    temperature_c=24
)
```

必须验证：

```text
18 <= temperature_c <= 30
```

对于未来可能涉及安全关键操作的能力：

```text
steering
brake
throttle
```

本项目默认：

```text
DISABLED
```

不允许通过自然语言 Agent 直接控制。

---

# 11. MCP Server

MCP Server 至少暴露：

```text
vehicle.get_status
vehicle.set_temperature

navigation.get_status
navigation.search
navigation.start

media.play
media.pause
media.volume

environment.get_context
```

每个 Tool 必须包含：

```text
name
description
inputSchema
outputSchema
```

工具描述必须具体，避免：

```text
do_vehicle_action()
```

应该：

```text
set_cabin_temperature()
```

输入：

```json
{
  "temperature_c": 24.0
}
```

MCP 工具发现与调用应遵循当前 MCP 规范，不要自定义一套“伪 MCP”协议。

---

# 12. Memory Architecture

不要只做 Conversation Summary。

必须至少分成：

```text
Working Memory
Episodic Memory
Semantic Memory
Preference Memory
```

## Preference Memory

例如：

```json
{
  "subject": "user",
  "attribute": "preferred_temperature",
  "value": 24,
  "unit": "celsius",
  "confidence": 0.92,
  "source": "conversation",
  "valid_from": "2026-08-01",
  "valid_to": null
}
```

## Episodic Memory

```json
{
  "event": "went_to_gym",
  "location": "Gym",
  "time": "2026-08-28T19:00:00",
  "participants": ["user"]
}
```

## Temporal Rule

新偏好不能简单 append。

例如：

```text
2025:
preferred_temperature = 25

2026:
preferred_temperature = 23
```

Retrieval 时：

```text
latest valid preference
>
historical preference
```

---

# 13. Memory Extraction

对每轮重要对话执行：

```text
Conversation
 ↓
Memory Candidate Extraction
 ↓
Candidate Validation
 ↓
Deduplication
 ↓
Conflict Resolution
 ↓
Persist
```

不是每句话都写 Memory。

必须判断：

```text
Is this durable?
Is this user-specific?
Is this useful later?
Is confidence sufficient?
Does it conflict with existing memory?
```

例如：

> “今天我有点冷。”

默认：

```text
NOT durable memory
```

而：

> “以后冬天车里保持 24℃比较舒服。”

应该成为：

```text
Preference Memory
```

---

# 14. Context Aggregation

Agent 每次执行前生成：

```json
{
  "user": {},
  "vehicle": {},
  "navigation": {},
  "environment": {},
  "memory": {},
  "conversation": {}
}
```

Context 优先级：

```text
Safety / Policy
>
Current Vehicle State
>
Current Environment
>
Recent Conversation
>
Latest User Preference
>
Historical Memory
```

不要把全部数据库 Memory 原样塞进 Prompt。

必须先 Retrieval。

---

# 15. Agent Graph

推荐：

```text
START
  ↓
load_context
  ↓
retrieve_memory
  ↓
classify_intent
  ↓
plan
  ↓
policy_check
  ↓
tool_execution
  ↓
observe_result
  ↓
need_more_action?
 ├── YES → plan
 └── NO
       ↓
generate_response
       ↓
extract_memory
       ↓
END
```

LangGraph 的动态 Agent 模式适合这种需要根据 Tool 结果决定下一步的场景；其 runtime 也支持状态持久化和长流程执行。

---

# 16. Planner

Planner 输出结构化计划：

```json
{
  "goal": "go_to_office_and_play_music",
  "steps": [
    {
      "id": 1,
      "tool": "navigation.search_destination",
      "arguments": {
        "query": "office"
      }
    },
    {
      "id": 2,
      "tool": "navigation.start_navigation",
      "arguments": {}
    },
    {
      "id": 3,
      "tool": "media.play",
      "arguments": {
        "playlist": "user_work_playlist"
      }
    }
  ]
}
```

禁止让 LLM 输出任意 Python 代码执行。

---

# 17. Multimodal Context

第一阶段：

```text
CARLA Camera
 ↓
image
 ↓
VLM
 ↓
structured scene
```

输出：

```json
{
  "weather": "rain",
  "traffic_density": "high",
  "road_work": true,
  "pedestrians": true,
  "confidence": 0.87
}
```

Agent 使用结构化结果，而不是每次把大量图片直接塞入主 Agent Prompt。

---

# 18. Evaluation

必须建立自己的 Automotive Agent Benchmark。

至少 100 个任务。

分类：

```text
vehicle_query
vehicle_control
navigation
media
memory
multi_step
multimodal
ambiguous
unsafe_request
```

每个 Case：

```json
{
  "id": "nav_001",
  "user": "帮我导航到公司",
  "expected_tools": [
    "navigation.search_destination",
    "navigation.start_navigation"
  ],
  "expected_success": true
}
```

---

# 19. Metrics

必须实现：

## Intent Accuracy

```text
正确 Intent / 总任务
```

## Tool Selection Accuracy

```text
正确 Tool / Tool Calls
```

## Argument Accuracy

```text
正确参数 / 参数总数
```

## Task Success Rate

```text
成功完成任务 / 总任务
```

## Memory Retrieval Precision

```text
Relevant Retrieved Memory / Retrieved Memory
```

## Memory Retrieval Recall

```text
Relevant Retrieved Memory / Relevant Memory
```

## Hallucination Rate

统计 Agent：

```text
不存在的 Tool
错误的 Tool
虚构的执行结果
```

## Latency

记录：

```text
P50
P95
P99
```

---

# 20. Observability

每次 Agent Run 必须记录：

```text
request_id
session_id
user_id
model
prompt_version

input
context

plan
tool_calls
tool_results

latency
token_usage

memory_reads
memory_writes

final_response
error
```

建议建立 Trace：

```text
Agent Run
 ├── Context
 ├── Memory Retrieval
 ├── LLM Call
 ├── Tool Call
 ├── Tool Result
 ├── LLM Call
 └── Final Response
```

---

# 21. API

FastAPI 至少提供：

```http
POST /api/v1/agent/chat
```

Request：

```json
{
  "user_id": "demo-user",
  "session_id": "session-001",
  "message": "有点冷"
}
```

Response：

```json
{
  "session_id": "session-001",
  "response": "我帮你把车内温度调到24℃。",
  "tool_calls": [
    {
      "name": "set_temperature",
      "arguments": {
        "temperature_c": 24
      }
    }
  ]
}
```

---

# 22. CLI Demo

必须提供：

```bash
python -m app.cli
```

交互：

```text
You > 有点冷

Agent > 当前车内21℃。
        根据你的偏好，我帮你调到24℃。

Tool > set_temperature(24)

You > 帮我导航去公司

Agent > 已找到公司位置，正在开始导航。
```

---

# 23. CARLA Integration

第一阶段使用：

```text
CARLA Python API
```

实现：

```python
CarlaClient
VehicleAdapter
WorldManager
SensorManager
```

启动流程：

```text
CARLA Server
 ↓
CarlaClient
 ↓
VehicleAdapter
 ↓
VehicleService
 ↓
Agent Tool
```

不要让业务层 import `carla`。

---

# 24. ROS2 第二阶段

完成 MVP 后再接：

```text
CARLA
 ↕
CARLA ROS Bridge
 ↕
ROS2
 ↕
Automotive Agent
```

ROS bridge 的官方实现提供 CARLA 与 ROS/ROS2 之间的双向数据转换，因此适合作为后续工程化阶段，而不是 MVP 的硬依赖。

---

# 25. Docker

Docker Compose 至少管理：

```text
api
postgres
redis
observability
```

CARLA 建议作为独立运行环境，根据宿主机 GPU/图形环境配置。

不要强制把 CARLA、Agent API、数据库全部打成一个镜像。

---

# 26. Configuration

所有环境变量必须来自 `.env`：

```text
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=

VLM_BASE_URL=
VLM_MODEL=

POSTGRES_URL=
REDIS_URL=

CARLA_HOST=
CARLA_PORT=

MCP_HOST=
MCP_PORT=
```

禁止代码中硬编码 API Key。

---

# 27. Implementation Phases

## Phase 0：Project Bootstrap

完成：

- Python project
- FastAPI
- Pydantic
- pytest
- Docker Compose
- configuration

验收：

```bash
pytest
```

必须通过。

---

## Phase 1：Vehicle Simulator

实现：

```text
CARLA connection
VehicleAdapter
VehicleState
Vehicle Tools
```

验收：

```text
get_vehicle_status()
set_temperature()
```

能够真实影响仿真状态。

---

## Phase 2：Basic Agent

实现：

```text
LangGraph
LLM
Tool Calling
```

验收：

```text
“现在车速多少？”
“把温度调到24度。”
```

成功执行。

---

## Phase 3：Memory

实现：

```text
Preference Memory
Episodic Memory
Retrieval
Conflict Resolution
```

验收：

```text
第一次：
“我喜欢车内24℃。”

第二次：
“有点冷。”

Agent:
set_temperature(24)
```

---

## Phase 4：Planner

支持多步任务：

```text
导航 + 媒体
```

验收：

```text
一个用户请求
→ >= 2 个 Tool Calls
→ 正确依赖顺序
```

---

## Phase 5：MCP

实现 Automotive MCP Server。

验收：

```text
MCP Client
→ tools/list
→ tool call
→ CARLA
```

---

## Phase 6：Multimodal

实现：

```text
CARLA Camera
→ VLM
→ Environment Context
→ Agent
```

---

## Phase 7：Evaluation

实现：

```text
100+ scenarios
自动执行
指标统计
JSON/CSV report
```

---

## Phase 8：Observability

实现：

```text
Trace
Metrics
Logs
Dashboard
```

---

# 28. AI Coding 执行规则

AI Coding Agent 必须遵守：

### Rule 1

先阅读：

```text
SPEC.md
```

然后再修改代码。

### Rule 2

每完成一个 Phase：

```text
implement
→ test
→ fix
→ document
```

### Rule 3

禁止一次性生成整个项目。

采用：

```text
Phase-by-Phase
```

### Rule 4

每个新模块必须有：

```text
unit test
```

### Rule 5

任何 CARLA、LLM、数据库依赖必须提供 mock。

这样单元测试不依赖 CARLA Server。

### Rule 6

LLM 调用必须可替换。

实现：

```text
LLMProvider
 ├── OpenAICompatibleProvider
 ├── LocalQwenProvider
 └── MockLLMProvider
```

### Rule 7

Agent 不得直接操作数据库。

必须：

```text
Agent
→ Service
→ Repository
→ Database
```

### Rule 8

Agent 不得直接 import CARLA。

必须：

```text
Agent
→ Tool
→ VehicleService
→ CarlaAdapter
→ CARLA
```

---

# 29. Definition of Done

项目达到以下条件才算完成 MVP：

- [ ] CARLA 可以正常启动
- [ ] Agent 可以读取 Vehicle State
- [ ] Agent 可以调用 Vehicle Tool
- [ ] Agent 可以执行多步任务
- [ ] Memory 可以跨 Session 工作
- [ ] Preference Conflict 可以处理
- [ ] MCP Server 可以暴露 Vehicle Tools
- [ ] Camera 可以生成 Environment Context
- [ ] 至少 100 个 Evaluation Cases
- [ ] 有 Task Success Rate
- [ ] 有 Tool Accuracy
- [ ] 有 Memory Precision / Recall
- [ ] 有 P50/P95 latency
- [ ] 有完整 Trace
- [ ] 有 Docker Compose
- [ ] 有 README Demo
- [ ] 有架构图
- [ ] 有测试
- [ ] 没有硬编码 API Key
- [ ] 安全关键车辆控制默认关闭

---

# 30. 求职作品集要求

README 首页必须首先展示：

```text
Architecture Diagram
Demo GIF / Video
Benchmark Result
Tech Stack
Quick Start
```

不要首先写几十页技术原理。

README 必须能够让面试官在 3 分钟内理解：

```text
What?
Why?
Architecture?
How?
Result?
```

---

# 31. 简历描述

最终项目完成后，可以使用类似：

> **Multimodal Personalized Automotive Agent**
>
> Built a CARLA-based automotive Agent integrating LangGraph, MCP, multimodal scene understanding, personalized long-term memory and vehicle/navigation/media tools; implemented stateful multi-step planning, temporal preference memory, tool safety policies and an automated benchmark covering task success, tool selection, memory retrieval and latency.

中文：

> **多模态个性化车载 Agent**
>
> 基于 CARLA 构建车载 Agent 仿真系统，使用 LangGraph 实现有状态 Agent 编排，集成 MCP Tool Calling、车辆/导航/媒体工具、多模态环境理解及长期个性化 Memory；设计时序偏好记忆、工具安全策略和 100+ 场景自动化 Benchmark，从任务成功率、Tool Selection、Memory Retrieval、Latency 等维度评估 Agent。

---

# 32. 最终技术路线

```text
                 ┌─────────────────────┐
                 │        User         │
                 └──────────┬──────────┘
                            ↓
                    Voice / Text / Image
                            ↓
                ┌───────────────────────┐
                │   Context Aggregator  │
                └──────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   CARLA State         Camera/VLM          Memory
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                  ┌─────────────────┐
                  │   LangGraph     │
                  │ AutomotiveAgent │
                  └────────┬────────┘
                           ↓
                  Planning / Policy
                           ↓
                       MCP Tools
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
     Vehicle           Navigation           Media
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                         CARLA

             ┌─────────────────────────┐
             │ Observability / Eval    │
             └─────────────────────────┘
```

---

## 33. 开发优先级

如果时间有限，严格按：

```text
P0  CARLA + Vehicle Tool
P0  LLM Agent
P0  LangGraph
P0  Memory
P1  Planner
P1  MCP
P1  Evaluation
P2  Multimodal
P2  Observability
P3  ROS2
P3  Edge/Cloud
```

不要一开始就做 ROS2、VLM、Edge AI。

**先让 Agent 真正“能看懂车辆状态 → 调用工具 → 改变仿真世界 → 记住用户 → 完成多步任务”。**

这才是本项目最核心的可展示闭环。
