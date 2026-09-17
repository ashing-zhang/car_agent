# Evaluation Evaluator - 自动执行评估场景并收集结果(规格第19节,Phase 7)
# 运行指南:
#   from app.evaluation.evaluator import Evaluator, run_evaluation
#   results = Evaluator().run()
#   report = run_evaluation()
# 运行环境:
#   - 必须配置真实 LLM API Key(.env 中 DASHSCOPE_API_KEY / LLM_API_KEY)
#   - 多工具场景走 LLMPlanner(含规则 Planner fallback),单工具场景走 ReAct Agent

import logging
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.graph import build_agent
from app.agent.llm_planner import LLMPlanner
from app.agent.plan_graph import PlanExecuteState, build_plan_agent
from app.agent.tool_registry import ToolRegistry
from app.config import AppYamlConfig, EvalYamlConfig, get_app_config, get_eval_config
from app.evaluation.dataset import EvalCase, load_scenarios
from app.evaluation.metrics import CaseResult, MetricsReport, compute_metrics
from app.llm.provider import get_llm_provider
from app.memory.extractor import MemoryExtractor
from app.memory.repository import InMemoryMemoryRepository
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


class Evaluator:
    """评估器:使用真实 LLM + LLMPlanner(规则 fallback)对每个场景执行并记录。"""

    def __init__(
        self,
        app_config: AppYamlConfig | None = None,
        eval_config: EvalYamlConfig | None = None,
    ) -> None:
        """加载应用与评估配置,并在初始化阶段预检 LLM。"""
        self._config: AppYamlConfig = app_config or get_app_config()
        self._eval_cfg: EvalYamlConfig = eval_config or get_eval_config()
        self._llm_prototype = get_llm_provider()
        logger.info("Evaluator initialized with real LLM (%s)", type(self._llm_prototype).__name__)

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
        """构造隔离的 ReAct Agent(独立 memory + 真实 LLM + 全工具 registry)。"""
        repo = InMemoryMemoryRepository()
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

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        """执行单个场景并返回结果;多工具走 Plan-Execute,单工具走 ReAct。"""
        start = time.perf_counter()
        actual_tools: list[str] = []
        actual_args: list[dict] = []

        use_plan = len(case.expected_tools) > 1
        try:
            if use_plan:
                plan_agent, _ = self._make_plan_agent()
                state: PlanExecuteState = plan_agent.invoke(
                    {
                        "messages": [HumanMessage(content=case.user)],
                        "user_id": "eval-user",
                        "session_id": f"eval-plan-{case.id}",
                    }
                )
                for tc in state.get("tool_calls", []):
                    actual_tools.append(tc.name)
                    actual_args.append(tc.arguments)
            else:
                agent = self._make_react_agent()
                result: dict[str, list[BaseMessage]] = agent.invoke(
                    {
                        "messages": [HumanMessage(content=case.user)],
                        "user_id": "eval-user",
                        "session_id": f"eval-react-{case.id}",
                    }
                )
                for msg in result.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            actual_tools.append(tc["name"])
                            actual_args.append(tc["args"])
        except Exception:
            logger.exception("Case %s execution raised", case.id)
            raise

        latency = (time.perf_counter() - start) * 1000
        expected_set = set(case.expected_tools)
        actual_set = set(actual_tools)

        if case.expected_reject:
            success = not (actual_set & SAFETY_CRITICAL_TOOLS)
        elif expected_set:
            success = expected_set.issubset(actual_set)
        else:
            success = True

        return CaseResult(
            case=case,
            actual_tools=actual_tools,
            actual_arguments=actual_args,
            success=success,
            latency_ms=latency,
        )

    def run(self, cases: list[EvalCase] | None = None) -> list[CaseResult]:
        """运行全部场景,异常不会中断整体评估。"""
        cases = cases or load_scenarios()
        results: list[CaseResult] = []
        for case in cases:
            try:
                results.append(self.evaluate_case(case))
            except Exception:
                logger.exception("Case %s failed", case.id)
                results.append(CaseResult(case=case, success=False, latency_ms=0.0))
        return results


def run_evaluation(cases: list[EvalCase] | None = None) -> MetricsReport:
    """便捷入口:运行评估并返回指标报告。"""
    evaluator = Evaluator()
    results = evaluator.run(cases)
    return compute_metrics(results)
