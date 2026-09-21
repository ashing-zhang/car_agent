个性化车载 Agent（AutoAgent）
技术栈：Python / LangGraph / LangChain Tools / FastAPI / PostgreSQL + pgvector / Pydantic v2 / Vue 3 / FastMCP / Prometheus / Docker / pytest
项目背景：面向智能座舱场景，构建以场景池驱动、支持长期记忆与多步规划的个性化 Automotive Agent，验证"多模态感知 → 记忆检索 → 意图分类路由 → 规划 → 安全校验 → Tool 执行 → 环境闭环 → 偏好写入"核心链路。

核心贡献：

1. Orchestrator 统一入口 + 双通道 Agent + DDD 分层架构
基于 LangGraph 构建 ReAct（单步 Tool Calling）+ PlanGraph（多步计划执行）双通道 Agent；FastAPI 暴露统一入口 `POST /api/v1/agent/chat`，由 `AgentOrchestrator` 通过意图分类器（规则分类器优先+LLM分类器兜底+置信度阈值降级）自动路由 ReAct/Plan-Execute，屏蔽单步/多步差异；附加 /health 健康检查、/vehicle/* 状态与控制、/observability/* 指标面板与 GET /metrics Prometheus 抓取端点共四类 REST 能力；严格遵循 DDD 分层：Agent Tool → VehicleService（Policy 校验 + 偏好自动同步）→ VehicleProvider Port → SceneVehicleProvider（委托模式）→ VehicleStateStore（会话级仓储 Protocol，开闭设计 + in-memory 后端实现，预留 Redis 扩展接口）+ YAML 场景池，解耦业务与数据源；修复 PlanExecuteState.plan 字段 Reducer 从 operator.add（追加）改为默认 replace，避免 execute 循环中 plan 重复累加导致 search/start 等工具被重复执行。

2. 15 项类型化 LangChain @tool + VehicleService 扩展能力 + 车速安全守卫
ToolRegistry 注册 15 项 Agent 可直接调用的类型化 @tool：①车辆类 4 项（get_vehicle_status/get_cabin_temperature/set_temperature/set_ac）②导航类 4 项（get_navigation_status/search_destination/start_navigation/cancel_navigation）③媒体类 3 项（play_media/pause_media/set_volume）④环境与多模态感知类 3 项（get_weather/get_traffic/get_camera_scene）；VehicleService 层另通过 VEHICLE_TOOL_DEFINITIONS 预留 8 项结构化校验的座舱控制方法（座椅位置/通风/按摩/加热，驾驶位+副驾独立 0–3 档、四车窗开度、后备箱、雨刮 5 档、车灯 5 模式），可按需挂载或经 MCP 协议对外暴露；PolicyConfig 统一参数范围校验，set_window 车速 > 120 km/h 禁开窗、set_trunk 车速 > 5 km/h 禁后备箱，越界请求返回结构化错误码并告警，确保座舱安全。

3. 时序偏好记忆 + 规则 Planner 模糊语义温度映射 + 自动弱同步闭环
MemoryService 分层架构：Extractor（候选抽取）→ Retriever（偏好检索 + pgvector 语义相似度）→ Repository（in-memory/PostgreSQL，通过 build_repository 工厂切换）→ EmbeddingProvider，支持 confidence/valid_from/valid_to 三时序属性与 latest_wins 冲突解决策略；规则 Planner `_extract_temperature` 新增模糊语义温度映射（"有点冷"→24°C、"太热了"→22°C），覆盖口语化非数字化温度表达；set_temperature/set_ac 执行成功后自动写入低置信度（默认 0.55）偏好，会话初始化时偏好叠加至场景状态，实现"用户习惯→场景默认值→Agent 决策"自动化闭环。

4. LLMPlanner + 规则 Planner 双层鲁棒性增强（7 项 Bad Case 专项治理 100% 修复）
LLMPlanner System Prompt 引入三大强约束条款：①【强制导航两步走】涉及导航必须先生成 search_destination 再紧跟 start_navigation，即使搜索结果显而易见也不可省略；②【POI 启发式】"家/公司/学校/商场/医院/机场/火车站" 等典型 POI 识别为目的地（例"回家"中"家"即目的地），避免误判为媒体歌单名；③【多意图顺序】并列意图先完成完整导航两步再追加媒体等其他步骤；新增静态校验 `LLMPlanner._validate_navigation_integrity`，检测 search 后未紧跟 start 即带错误反馈重试（最多 2 次），耗尽后自动降级至规则 Planner。规则 Planner `_extract_destination` 配置 17 项停止词表（顺便/然后/同时/路上/途中/的路上/的时候/并/且/和/跟/还有/再/还/逗号/句号/问号等），按最早出现的停止词精确截断，解决"去X顺便放Y"/"回家路上播放音乐"场景目的地提取错误；LLMPlanner 与规则 Planner 均通过 test_bad_case_fixes 单测覆盖 ms_001/003/004/006 多步类场景、mem_003/004 记忆类场景、mm_001 多模态类场景。

5. 全链路可观测 + 离线评估（PostgreSQL 仓储，开闭原则工厂）+ 9 大类 88 个场景评估体系
自建 TraceRecorder（request_id 追踪 + Span 分段 + latency_ms 计时）与 MetricsRegistry（Counter/Histogram 两类指标，含 agent_runs、tool_calls、延迟分布直方图），结构化 JSON 日志贯穿全链路；Evaluator 离线评估框架遵循开闭原则，通过 `build_repository("postgres")` 工厂函数创建 PostgreSQL 记忆仓储（替代硬编码 InMemoryMemoryRepository），单工具场景走 ReAct + MockLLMProvider，多工具场景走 PlanGraph，覆盖 9 大类 81 个场景（ambiguous4/media7/memory7/multi_step6/multimodal6/navigation12/unsafe_request5/vehicle_control10/vehicle_query18）+ 7 项 Bad Case 专项复评，共 88 case；计算成功率/工具选择准确率/参数准确率/意图分类准确率/幻觉率/P50/P95/P99 等 8 项指标；7 项 Bad Case 复评（记忆类 2 + 多步类 4 + 多模态类 1）全部通过，fix_rate=1.0，意图分类准确率 100%、工具选择准确率 100%、参数准确率 100%、任务成功率 100%、幻觉率 0%，延迟 P50 2.58s / P95 3.97s / P99 4.32s；配套 Prometheus exposition 文本渲染器暴露 /metrics 端点供抓取；Vue 3 + Vite 三 Tab 前端：智能助手对话/车辆控制可视化/可观测面板。

6. MCP Server + CLI + 配置解耦与生产级工程实践
FastMCP 实现 Automotive MCP Server，按 MCP 规范以 vehicle.* / navigation.* / media.* 命名空间对外暴露工具（`python -m app.mcp.server` stdio 模式启动），支持跨客户端工具发现与调用；`python -m app.cli` 提供交互式 REPL 命令行演示入口；敏感配置走 Pydantic Settings（.env），非敏感运行配置走 YAML + AppYamlConfig（LLM/Agent/Memory/IntentClassifier/Policy/Observability/Simulation 七类）；最新 Python 语法（| 联合类型、无冗余 __init__.py）、委托模式替代继承、全函数参数与返回值类型化、单测覆盖（test_vehicle_tools/test_memory/test_evaluation/test_bad_case_fixes/test_evaluator_pg_repo/test_intent_classifier/test_mcp 等）；Dockerfile 生产镜像 COPY 代码打包 + docker-compose 开发环境卷挂载。

项目总结：通过场景池 + DDD 分层 + Protocol 开闭设计将车载 Agent 从"LLM 对话 Demo"升级为生产级闭环系统，核心解决 15 项类型化 Tool Calling + 8 项座舱控制扩展能力与车速安全守卫、Orchestrator 意图分类路由双通道任务规划（导航两步走静态校验 + 规则降级）、偏好记忆自动同步（含模糊口语温度映射）、7 项 Bad Case 专项治理（100% 修复覆盖记忆/多步/多模态三类）、离线评估（build_repository 工厂 + PostgreSQL 仓储，9 大类 88 case）、全链路可观测（Trace/Span/Prometheus 端点）、MCP 标准协议对外暴露等关键问题，形成"场景池 → 记忆检索 → LangGraph 双通道（Orchestrator 自动路由）→ Policy 安全校验 → 15 工具执行（+8 项预留扩展）→ 环境闭环 → 偏好写入 → Evaluator（Bad Case 复评）+ Prometheus 指标"完整技术闭环，为接入 ROS 2/真实车载 API/CARLA 仿真等 Automotive AI 技术栈预留平滑扩展能力。
