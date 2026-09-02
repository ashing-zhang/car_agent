# Evaluation Evaluator - 自动执行评估场景并收集结果(规格第19节,Phase 7)
# 运行指南:
#   from app.evaluation.evaluator import Evaluator, run_evaluation
#   results = Evaluator().run()
#   report = run_evaluation()
# 多步场景用 plan-execute,单工具场景用 ReAct(MockLLM),不依赖外部 API

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent
from app.agent.plan_graph import run_plan
from app.agent.tool_registry import ToolRegistry
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.config import get_app_config
from app.evaluation.dataset import EvalCase, load_scenarios
from app.evaluation.metrics import CaseResult, MetricsReport, compute_metrics
from app.llm.provider import MockLLMProvider
from app.memory.extractor import MemoryExtractor
from app.memory.repository import InMemoryMemoryRepository
from app.memory.retriever import MemoryRetriever
from app.memory.service import MemoryService
from app.tools.environment import get_environment_service
from app.tools.media import get_media_service
from app.tools.navigation import get_navigation_service
from app.tools.vehicle import VehicleService

logger = logging.getLogger(__name__)

SAFETY_CRITICAL_TOOLS = {"steering", "brake", "throttle"}


class Evaluator:
    """评估器:对每个场景选择合适 Agent 并记录工具调用与延迟。"""

    def __init__(self) -> None:
        """加载应用配置。"""
        self._config = get_app_config()

    def _make_react_agent(self) -> object:
        """构造隔离的 ReAct Agent(独立 memory + MockLLM + 全工具 registry)。"""
        repo = InMemoryMemoryRepository()
        mem = MemoryService(
            MemoryExtractor(self._config.memory.min_confidence),
            MemoryRetriever(repo, self._config.memory.retrieval_top_k),
            repo,
        )
        service = VehicleService(SceneVehicleProvider(get_scene_pool()), self._config.policy)
        llm = MockLLMProvider()
        registry = ToolRegistry(
            service, get_navigation_service(), get_media_service(), get_environment_service()
        )
        return build_agent(service, llm, memory_service=mem, registry=registry)

    def evaluate_case(self, case: EvalCase) -> CaseResult:
        """执行单个场景并返回结果。"""
        start = time.perf_counter()
        actual_tools: list[str] = []
        actual_args: list[dict] = []

        use_plan = len(case.expected_tools) > 1
        if use_plan:
            state = run_plan(case.user, "eval-user")
            for tc in state.get("tool_calls", []):
                actual_tools.append(tc.name)
                actual_args.append(tc.arguments)
        else:
            agent = self._make_react_agent()
            result = agent.invoke(
                {"messages": [HumanMessage(content=case.user)], "user_id": "eval-user", "session_id": "eval"}
            )
            for msg in result.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        actual_tools.append(tc["name"])
                        actual_args.append(tc["args"])

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
        """运行全部场景。"""
        cases = cases or load_scenarios()
        results: list[CaseResult] = []
        for case in cases:
            try:
                results.append(self.evaluate_case(case))
            except Exception as exc:
                logger.exception("Case %s failed: %s", case.id, exc)
                results.append(CaseResult(case=case, success=False, latency_ms=0.0))
        return results


def run_evaluation(cases: list[EvalCase] | None = None) -> MetricsReport:
    """便捷入口:运行评估并返回指标报告。"""
    evaluator = Evaluator()
    results = evaluator.run(cases)
    return compute_metrics(results)
