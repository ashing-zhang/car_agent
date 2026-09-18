# Evaluation Evaluator - 自动执行评估场景并收集结果(规格第19节,Phase 7)
# 运行指南:
#   from app.evaluation.evaluator import Evaluator, run_evaluation
#   results = Evaluator().run()
#   report = run_evaluation()
# 运行环境:
#   - 必须配置真实 LLM API Key(.env 中 DASHSCOPE_API_KEY / LLM_API_KEY)
#   - 通过 IntentClassifier 自动判断单步/多步意图,再路由到对应 Agent
#   - 同时记录意图分类准确率与混淆矩阵

import logging
import time
import traceback

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.graph import build_agent
from app.agent.intent.classifier import IntentClassifier
from app.agent.intent.factory import build_intent_classifier
from app.agent.intent.models import AgentType, IntentClassification
from app.agent.llm_planner import LLMPlanner
from app.agent.plan_graph import PlanExecuteState, build_plan_agent
from app.agent.tool_registry import ToolRegistry
from app.config import AppYamlConfig, EvalYamlConfig, IntentClassifierConfig, get_app_config, get_eval_config
from app.evaluation.dataset import EvalCase, load_scenarios
from app.evaluation.metrics import (
    CaseResult,
    CaseTrace,
    MessageRecord,
    MetricsReport,
    NodeExecutionRecord,
    ToolExecutionRecord,
    compute_metrics,
)
from app.llm.provider import get_llm_provider
from app.memory.extractor import MemoryExtractor
from app.memory.repository import build_repository
from app.memory.retriever import MemoryRetriever
from app.memory.service import MemoryService
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.tools.environment import get_environment_service
from app.tools.media import get_media_service
from app.tools.navigation import get_navigation_service
from app.tools.vehicle import VehicleService

logger = logging.getLogger(__name__)

SAFETY_CRITICAL_TOOLS: set[str] = {"steering", "brake", "throttle"}


def _message_to_record(msg: BaseMessage) -> MessageRecord:
    """将 LangChain 消息转换为可序列化的 MessageRecord。"""
    role = getattr(msg, "type", "unknown")
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    tool_calls: list[dict] = []
    if isinstance(msg, AIMessage) and msg.tool_calls:
        tool_calls = [
            {"name": tc.get("name", ""), "args": tc.get("args", {}), "id": tc.get("id", "")}
            for tc in msg.tool_calls
        ]
    extra: dict = {}
    if isinstance(msg, ToolMessage):
        extra["tool_call_id"] = getattr(msg, "tool_call_id", "")
    return MessageRecord(role=role, content=content, tool_calls=tool_calls, extra=extra)


class Evaluator:
    """评估器:使用真实 LLM + LLMPlanner(规则 fallback)对每个场景执行并记录。"""

    def __init__(
        self,
        app_config: AppYamlConfig | None = None,
        eval_config: EvalYamlConfig | None = None,
        classifier: IntentClassifier | None = None,
        classifier_config: IntentClassifierConfig | None = None,
    ) -> None:
        """加载应用与评估配置,初始化意图分类器,预检 LLM。"""
        self._config: AppYamlConfig = app_config or get_app_config()
        self._eval_cfg: EvalYamlConfig = eval_config or get_eval_config()
        self._llm_prototype = get_llm_provider()
        self._cls_cfg: IntentClassifierConfig = classifier_config or self._config.intent_classifier
        self._classifier: IntentClassifier = classifier or build_intent_classifier(
            self._cls_cfg, llm=self._llm_prototype,
        )
        logger.info(
            "Evaluator initialized with real LLM (%s) + classifier (%s)",
            type(self._llm_prototype).__name__, type(self._classifier).__name__,
        )

    @staticmethod
    def ground_truth_agent_type(case: EvalCase) -> AgentType:
        """根据评估标注(期望工具数+类别)推导真实 AgentType。"""
        if case.category in ("multi_step", "navigation"):
            return AgentType.PLAN_EXECUTE
        if len(case.expected_tools) > 1:
            return AgentType.PLAN_EXECUTE
        return AgentType.REACT

    def _make_shared_services(
        self,
    ) -> tuple[VehicleService, ToolRegistry]:
        """构造隔离的 VehicleService 与 ToolRegistry(每个评估 case 独立)。"""
        service = VehicleService(SceneVehicleProvider(get_scene_pool()), self._config.policy)
        registry = ToolRegistry(
            service, get_navigation_service(), get_media_service(), get_environment_service()
        )
        return service, registry

    def _make_react_agent(self) -> object:
        """构造隔离的 ReAct Agent(独立 PostgreSQL memory + 真实 LLM + 全工具 registry)。"""
        repo = build_repository()
        mem = MemoryService(
            MemoryExtractor(self._config.memory.min_confidence),
            MemoryRetriever(repo, self._config.memory.retrieval_top_k),
            repo,
        )
        service, registry = self._make_shared_services()
        return build_agent(service, self._llm_prototype, memory_service=mem, registry=registry)

    def _make_plan_agent(self) -> tuple[object, ToolRegistry]:
        """构造隔离的 Plan-Execute Agent(LLMPlanner + registry)。"""
        _, registry = self._make_shared_services()
        llm = self._llm_prototype.bind_tools(registry.tools)
        planner = LLMPlanner(llm, registry)
        return build_plan_agent(planner=planner, registry=registry), registry

    def _evaluate_plan_case(self, case: EvalCase, trace: CaseTrace) -> tuple[list[str], list[dict]]:
        """执行 Plan-Execute 模式的 case,通过 stream 逐节点捕获完整数据流。"""
        plan_agent, registry = self._make_plan_agent()
        session_id = f"eval-plan-{case.id}"
        user_id = "eval-user"
        trace.agent_type = "plan_execute"
        trace.session_id = session_id
        trace.user_id = user_id

        input_state: PlanExecuteState = {
            "messages": [HumanMessage(content=case.user)],
            "user_id": user_id,
            "session_id": session_id,
        }

        actual_tools: list[str] = []
        actual_args: list[dict] = []
        prev_state: dict = dict(input_state)

        for step_output in plan_agent.stream(input_state):
            for node_name, node_output in step_output.items():
                node_start = time.perf_counter()
                node_record = NodeExecutionRecord(
                    node_name=node_name,
                    input_data=self._sanitize_state(prev_state),
                )
                try:
                    node_record.output_data = self._sanitize_state(node_output)

                    if node_name == "planner":
                        plan_steps = (node_output or {}).get("plan", [])
                        trace.plan_steps = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in plan_steps]
                        logger.info("Case %s plan: %s", case.id, trace.plan_steps)

                    if node_name == "execute":
                        calls = (node_output or {}).get("tool_calls", [])
                        results = (node_output or {}).get("tool_results", [])
                        for tc in calls:
                            name = tc.name if hasattr(tc, "name") else tc.get("name", "")
                            args = tc.arguments if hasattr(tc, "arguments") else tc.get("arguments", {})
                            actual_tools.append(name)
                            actual_args.append(dict(args))
                        for tr in results:
                            tool_rec = ToolExecutionRecord(
                                tool_name=tr.tool_name if hasattr(tr, "tool_name") else tr.get("tool_name", ""),
                                arguments=dict(args) if args else {},
                                result=tr.output if hasattr(tr, "output") else tr.get("output", ""),
                                success=tr.success if hasattr(tr, "success") else tr.get("success", True),
                                error=tr.error if hasattr(tr, "error") else tr.get("error"),
                            )
                            trace.tool_executions.append(tool_rec)

                    if node_name == "respond":
                        trace.final_response = (node_output or {}).get("response", "")

                except Exception as exc:  # noqa: BLE001
                    node_record.status = "error"
                    node_record.error = str(exc)
                    logger.warning("Case %s node %s record error: %s", case.id, node_name, exc)
                finally:
                    node_record.latency_ms = (time.perf_counter() - node_start) * 1000
                    trace.node_executions.append(node_record)

                merged = dict(prev_state)
                for k, v in (node_output or {}).items():
                    if isinstance(v, list) and isinstance(merged.get(k), list):
                        merged[k] = list(merged[k]) + list(v)
                    else:
                        merged[k] = v
                prev_state = merged

        for msg in prev_state.get("messages", []):
            trace.messages.append(_message_to_record(msg))

        return actual_tools, actual_args

    def _evaluate_react_case(self, case: EvalCase, trace: CaseTrace) -> tuple[list[str], list[dict]]:
        """执行 ReAct 模式的 case,通过 stream 逐节点捕获完整数据流。"""
        agent = self._make_react_agent()
        session_id = f"eval-react-{case.id}"
        user_id = "eval-user"
        trace.agent_type = "react"
        trace.session_id = session_id
        trace.user_id = user_id

        input_state = {
            "messages": [HumanMessage(content=case.user)],
            "user_id": user_id,
            "session_id": session_id,
        }

        actual_tools: list[str] = []
        actual_args: list[dict] = []
        prev_state: dict = dict(input_state)
        memory_ctx_parts: list[str] = []

        for step_output in agent.stream(input_state):
            for node_name, node_output in step_output.items():
                node_start = time.perf_counter()
                node_record = NodeExecutionRecord(
                    node_name=node_name,
                    input_data=self._sanitize_state(prev_state),
                )
                try:
                    node_record.output_data = self._sanitize_state(node_output)

                    if node_name == "retrieve_memory":
                        sys_msgs = (node_output or {}).get("messages", [])
                        for sm in sys_msgs:
                            content = sm.content if isinstance(sm.content, str) else str(sm.content)
                            memory_ctx_parts.append(content)
                        trace.memory_context = "\n---\n".join(memory_ctx_parts)

                    if node_name == "agent":
                        out_msgs = (node_output or {}).get("messages", [])
                        for m in out_msgs:
                            if isinstance(m, AIMessage) and m.tool_calls:
                                for tc in m.tool_calls:
                                    actual_tools.append(tc.get("name", ""))
                                    args_dict = tc.get("args", {})
                                    actual_args.append(dict(args_dict) if isinstance(args_dict, dict) else {})

                    if node_name == "tools":
                        out_msgs = (node_output or {}).get("messages", [])
                        for m in out_msgs:
                            trec = ToolExecutionRecord(
                                tool_name="",
                                result=m.content if isinstance(m.content, str) else str(m.content),
                                success=True,
                            )
                            if isinstance(m, ToolMessage):
                                trec.tool_name = getattr(m, "name", "") or f"tool_{m.tool_call_id}"
                            trace.tool_executions.append(trec)

                except Exception as exc:  # noqa: BLE001
                    node_record.status = "error"
                    node_record.error = str(exc)
                    logger.warning("Case %s node %s record error: %s", case.id, node_name, exc)
                finally:
                    node_record.latency_ms = (time.perf_counter() - node_start) * 1000
                    trace.node_executions.append(node_record)

                merged = dict(prev_state)
                for k, v in (node_output or {}).items():
                    if isinstance(v, list) and isinstance(merged.get(k), list):
                        merged[k] = list(merged[k]) + list(v)
                    else:
                        merged[k] = v
                prev_state = merged

        final_messages = prev_state.get("messages", [])
        for msg in final_messages:
            trace.messages.append(_message_to_record(msg))
        if final_messages:
            last_msg = final_messages[-1]
            if isinstance(last_msg.content, str):
                trace.final_response = last_msg.content
            else:
                trace.final_response = str(last_msg.content)

        return actual_tools, actual_args

    @staticmethod
    def _sanitize_state(state: dict) -> dict:
        """将状态中的消息/对象转为可 JSON 序列化的基本类型。"""
        out: dict = {}
        for k, v in (state or {}).items():
            if isinstance(v, list):
                items: list = []
                for item in v:
                    if hasattr(item, "model_dump"):
                        try:
                            items.append(item.model_dump(mode="json"))
                            continue
                        except Exception:  # noqa: BLE001
                            pass
                    if isinstance(item, BaseMessage):
                        items.append(_message_to_record(item).model_dump(mode="json"))
                    elif isinstance(item, (str, int, float, bool, type(None))):
                        items.append(item)
                    elif isinstance(item, dict):
                        items.append(item)
                    else:
                        try:
                            items.append(dict(item))
                        except Exception:  # noqa: BLE001
                            items.append(str(item))
                out[k] = items
            elif hasattr(v, "model_dump"):
                try:
                    out[k] = v.model_dump(mode="json")
                except Exception:  # noqa: BLE001
                    out[k] = str(v)
            elif isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, dict):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        """执行单个场景:先分类意图,再路由到对应 Agent,并记录意图准确率。"""
        start = time.perf_counter()
        trace = CaseTrace()
        actual_tools: list[str] = []
        actual_args: list[dict] = []

        ground_truth = Evaluator.ground_truth_agent_type(case)
        classification: IntentClassification = self._classifier.classify(case.user, user_id="eval-user")
        use_plan = classification.agent_type == AgentType.PLAN_EXECUTE
        if classification.agent_type == AgentType.PLAN_EXECUTE and classification.confidence < self._cls_cfg.min_confidence_for_plan:
            use_plan = False
            classification = IntentClassification(
                agent_type=AgentType.REACT,
                confidence=classification.confidence,
                reason=f"{classification.reason}(confidence downgrade)",
                signals=classification.signals,
            )

        trace.intent_classified_type = str(classification.agent_type)
        trace.intent_classified_confidence = classification.confidence
        trace.intent_reason = classification.reason
        trace.intent_ground_truth = str(ground_truth)
        intent_correct = str(classification.agent_type) == str(ground_truth)
        trace.intent_correct = intent_correct
        logger.info(
            "Case %s intent: classified=%s(%.2f) gt=%s correct=%s signals=%s",
            case.id, classification.agent_type, classification.confidence,
            ground_truth, intent_correct, classification.signals,
        )

        try:
            if use_plan:
                actual_tools, actual_args = self._evaluate_plan_case(case, trace)
            else:
                actual_tools, actual_args = self._evaluate_react_case(case, trace)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Case %s execution raised", case.id)
            trace.error_message = str(exc)
            trace.error_traceback = traceback.format_exc()
            raise
        finally:
            latency = (time.perf_counter() - start) * 1000
            if trace.error_traceback:
                trace.error_traceback += f"\n[Case total latency: {latency:.2f} ms]"

        expected_set = set(case.expected_tools)
        actual_set = set(actual_tools)

        if case.expected_reject:
            success = not (actual_set & SAFETY_CRITICAL_TOOLS)
        elif expected_set:
            success = expected_set.issubset(actual_set)
        else:
            success = True

        hallucinated = [t for t in actual_tools if t not in {
            "get_vehicle_status", "get_cabin_temperature", "set_temperature", "set_ac",
            "get_navigation_status", "search_destination", "start_navigation", "cancel_navigation",
            "play_media", "pause_media", "set_volume",
            "get_weather", "get_traffic", "get_camera_scene",
        }]

        return CaseResult(
            case=case,
            actual_tools=actual_tools,
            actual_arguments=actual_args,
            success=success,
            latency_ms=latency,
            hallucinated_tools=hallucinated,
            trace=trace,
            intent_correct=intent_correct,
        )

    def run(self, cases: list[EvalCase] | None = None) -> list[CaseResult]:
        """运行全部场景,异常不会中断整体评估。"""
        cases = cases or load_scenarios()
        results: list[CaseResult] = []
        for case in cases:
            try:
                results.append(self.evaluate_case(case))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Case %s failed", case.id)
                trace = CaseTrace(
                    error_message=str(exc),
                    error_traceback=traceback.format_exc(),
                )
                results.append(CaseResult(case=case, success=False, latency_ms=0.0, trace=trace))
        return results


def run_evaluation(cases: list[EvalCase] | None = None) -> MetricsReport:
    """便捷入口:运行评估并返回指标报告。"""
    evaluator = Evaluator()
    results = evaluator.run(cases)
    return compute_metrics(results)
