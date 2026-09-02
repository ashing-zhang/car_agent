# AutoAgent

> Multimodal Personalized Automotive Agent (Scene-Pool Driven Simulation)

多模态、个性化车载 Agent,车辆/环境仿真状态由场景池(configs/scene_pool.yaml)随机选景提供,无需部署 CARLA 等重型仿真器。基于 LangGraph 实现 Agent 编排,集成 MCP Tool Calling、车辆/导航/媒体工具及长期个性化 Memory。

## What

位于"用户 ↔ 智能座舱/车辆能力"之间的 Agent 层,通过自然语言理解用户意图,调用工具完成车辆控制、导航、媒体播放等任务,并跨会话记住用户偏好。

## Why

- 不是自动驾驶控制器,而是座舱 Agent 层
- 可展示 Agent 真正"感知车辆状态 → 调用工具 → 记住用户 → 完成多步任务"的闭环
- 场景池配置驱动,可扩展真实车机或其他仿真器
- 可替换 LLM,可扩展 ROS 2

## Architecture

```
User (Text/Voice)
    ↓
Context Aggregator
    ↓
LangGraph Agent (Planner / Router / Policy / Reflection)
    ↓
MCP Tools (Vehicle / Navigation / Media / Environment)
    ↓
Scene Pool (configs/scene_pool.yaml 随机选景)

Observability: Trace / Metrics / Logs / Evaluation
```

## Tech Stack

| Module | Tech |
|---|---|
| Language | Python 3.11+ |
| Agent Runtime | LangGraph |
| LLM | Qwen (DashScope OpenAI-compatible) |
| Simulation | Scene Pool (config-driven) |
| Backend | FastAPI |
| Frontend | Vue 3 + Vite |
| Protocol | MCP |
| Database | PostgreSQL + pgvector |
| Cache | Redis |
| ORM | SQLAlchemy |
| Validation | Pydantic |
| Testing | pytest |
| Container | Docker Compose |
| Observability | OpenTelemetry + Prometheus |

## Quick Start

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置环境变量(参考 .env.example)
cp .env.example .env

# 3. 启动基础设施
docker compose up -d postgres redis

# 4. 启动 API
python -m app.main

# 5. 启动前端 (Vue 3)
cd frontend && npm install && npm run dev
# 浏览器访问 http://localhost:5173
```

## Frontend (Vue 3)

正式前端界面替代 CLI,提供三个面板:

- **智能助手**:对话交互(单轮 ReAct / 多步 Plan),展示工具调用、request_id、延迟
- **车辆控制**:实时状态查询 + 温度策略校验
- **可观测性**:Trace 列表/详情 + 指标快照(对接 Phase 8)

开发模式由 Vite 将 `/api`、`/metrics` 代理到后端 `localhost:8000`;Docker 模式通过 `VITE_API_TARGET` 环境变量指向 `http://api:8000`。

## Demo

### 车辆状态查询
```
You > 现在车速多少?还有多少电?
Agent > 当前车速 63 km/h,电量 72%。
        Tool > get_vehicle_status()
```

### 个性化偏好
```
You > 我冬天喜欢车内保持 24℃。
# (Agent 写入 Preference Memory)

You > 有点冷。
Agent > 当前车内 21℃。根据你的偏好,我帮你调到 24℃。
        Tool > set_temperature(24)
```

### 多步任务
```
You > 我要去公司,顺便播放我平时上班喜欢听的音乐。
Agent > Plan: search navigation → start navigation → play music
```

## Phases

- [x] Phase 0: Project Bootstrap
- [ ] Phase 1: Vehicle Simulator
- [ ] Phase 2: Basic Agent
- [ ] Phase 3: Memory
- [ ] Phase 4: Planner
- [ ] Phase 5: MCP
- [ ] Phase 6: Multimodal
- [ ] Phase 7: Evaluation
- [ ] Phase 8: Observability

## API

```http
POST /api/v1/agent/chat
```

```json
Request:
{ "user_id": "demo-user", "session_id": "session-001", "message": "有点冷" }

Response:
{ "session_id": "session-001", "response": "我帮你把车内温度调到24℃。",
  "tool_calls": [{"name": "set_temperature", "arguments": {"temperature_c": 24}}] }
```

## Project Structure

```
autoagent/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── config.py        # 配置驱动
│   ├── api/             # API 路由
│   ├── agent/           # LangGraph Agent
│   ├── tools/           # Vehicle/Navigation/Media/Environment 工具
│   ├── mcp/             # MCP Server
│   ├── simulation/      # 场景池仿真(ScenePool/Providers)
│   ├── memory/          # 记忆系统
│   ├── context/         # Context Aggregator
│   ├── evaluation/      # Benchmark
│   └── observability/   # Trace/Metrics/Logs
├── tests/
├── configs/
├── frontend/         # Vue 3 前端
├── scripts/
└── docs/
```

## Safety

安全关键车辆控制(steering/brake/throttle)默认 DISABLED,不允许通过自然语言 Agent 直接控制。

## License

MIT
