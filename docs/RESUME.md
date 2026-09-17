个性化车载 Agent（AutoAgent）
技术栈：Python / LangGraph / LangChain Tools / FastAPI / PostgreSQL + pgvector / Redis / Pydantic v2 / Vue 3 / Docker / pytest
项目背景：面向智能座舱场景，构建以场景池驱动、支持长期记忆与多步规划的个性化 Automotive Agent，验证"感知 → 记忆检索 → 规划 → 安全校验 → Tool 执行 → 环境闭环"核心链路。

核心贡献：

1. 双通道 Agent 工作流 + DDD 分层架构
基于 LangGraph 构建 ReAct（单工具）+ PlanGraph（多步规划）双通道 Agent，FastAPI 提供 /chat 与 /plan 端点；严格遵循 DDD 分层：Agent Tool → VehicleService（Policy + 偏好同步）→ VehicleProvider Port → SceneVehicleProvider（委托模式）→ VehicleStateStore（会话级仓储，in-memory/Redis）+ YAML 场景池，解耦业务与数据源。

2. 12 项座舱工具集 + 车速安全守卫
基于 LangChain @tool 实现 12 项类型化工具：车辆状态查询、温度/空调、座椅（位置/通风/按摩/加热，驾驶位+副驾独立 0–3 档）、四车窗开度、后备箱、雨刮 5 档、车灯 5 模式；PolicyConfig 统一参数范围校验，车速 > 120 km/h 禁开窗、> 5 km/h 禁后备箱，越界请求返回结构化错误码并告警，确保座舱安全。

3. 时序偏好记忆 + 自动弱同步闭环
MemoryService 分层架构：Extractor（候选抽取）→ Retriever（偏好检索 + pgvector 语义）→ Repository（in-memory/PostgreSQL）→ EmbeddingProvider，支持 confidence / valid_from / valid_to 时序属性与 latest_wins 冲突解决；set_temperature / set_ac 成功后自动写入低置信度偏好，会话初始化时偏好叠加至场景状态，实现"用户习惯 → 场景默认值 → Agent 决策"自动化闭环。

4. 全链路可观测 + 离线评估 + Vue 3 交互面板
自建 TraceRecorder（request_id + Span + latency_ms）与 MetricsRegistry（Counter/Histogram，agent_runs、tool_calls、延迟直方图），结构化日志贯穿全链路；Evaluator 离线评估框架：单工具场景走 ReAct + MockLLMProvider，多工具走 PlanGraph，覆盖 8 类场景，计算成功率/工具选择准确率/参数准确率/P50/P95 等 7 项指标；Vue 3 + Vite 三 Tab 前端：智能助手对话/车辆控制可视化/可观测面板。

5. 配置解耦与生产级工程实践
敏感配置走 Pydantic Settings（.env），非敏感运行配置走 YAML + AppYamlConfig（LLM/Agent/Memory/Policy/Observability/Simulation 六类）；最新 Python 语法（| 联合类型、无冗余 __init__.py）、委托模式替代继承、全函数类型化、单测覆盖（test_vehicle_tools/test_memory/test_evaluation 等）；Dockerfile 生产镜像 COPY 打包 + docker-compose 开发卷挂载。

项目总结：通过场景池 + DDD 分层将车载 Agent 从"LLM 对话 Demo"升级为生产级闭环系统，核心解决 12 项座舱 Tool Calling 与车速安全守卫、双通道任务规划、偏好记忆自动同步、离线评估与全链路可观测等关键问题，形成"场景池 → 记忆检索 → LangGraph 双通道 → Policy 安全校验 → 12 工具执行 → 环境闭环 → 偏好写入 → Evaluator"完整技术闭环，为接入 ROS 2/真实车载 API/CARLA 仿真等 Automotive AI 技术栈预留平滑扩展能力。
