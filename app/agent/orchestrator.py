# Agent 统一编排器 Orchestrator - 意图分类后路由到 ReAct 或 Plan-Execute(配置驱动)
# 运行指南:
#   from app.agent.orchestrator import AgentOrchestrator, build_default_orchestrator
#   orch = build_default_orchestrator()
#   result = orch.run("去公司顺便播放音乐", user_id="u1")
#   # result.response / result.tool_calls / result.agent_type / result.classification
# 对外统一 API,内部根据意图分类路由

import logging
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.graph import build_agent as build_react_agent
from app.agent.intent.classifier import IntentClassifier
from app.agent.intent.factory import build_intent_classifier
from app.agent.intent.models import AgentType, IntentClassification
from app.agent.plan_graph import build_plan_agent
from app.agent.tool_registry import ToolRegistry, get_tool_registry
from app.config import IntentClassifierConfig, get_app_config
from app.llm.provider import LLMProvider, get_llm_provider
from app.tools.vehicle import VehicleService, get_vehicle_service

logger = logging.getLogger(__name__)


class OrchestratorResult(TypedDict, total=False):
    """编排器统一返回格式:屏蔽 ReAct/Plan-Execute 差异。"""

    response: str
    tool_calls: list[dict]
    agent_type: AgentType
    classification: IntentClassification
    raw_state: dict


def _normalize_react_output(state: dict) -> tuple[str, list[dict]]:
    """从 ReAct Agent 返回的 messages 中提取 response 与 tool_calls。"""
    messages: list[BaseMessage] = state.get("messages", [])
    tool_calls: list[dict] = []
    response = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc.get("name", ""), "arguments": tc.get("args", {})})
        if isinstance(msg, AIMessage) and msg.content and msg is messages[-1]:
            response = msg.content if isinstance(msg.content, str) else str(msg.content)
    return response, tool_calls


def _normalize_plan_output(state: dict) -> tuple[str, list[dict]]:
    """从 Plan-Execute Agent 返回的 state 中提取 response 与 tool_calls。"""
    tool_calls = [
        {"name": tc.name, "arguments": tc.arguments}
        for tc in state.get("tool_calls", [])
    ]
    response = state.get("response", "")
    return response, tool_calls


class AgentOrchestrator:
    """统一编排器:先意图分类,再路由到对应 Agent。支持依赖注入便于测试。"""

    def __init__(
        self,
        classifier: IntentClassifier,
        react_agent: object | None = None,
        plan_agent: object | None = None,
        service: VehicleService | None = None,
        llm_provider: LLMProvider | None = None,
        registry: ToolRegistry | None = None,
        classifier_config: IntentClassifierConfig | None = None,
    ) -> None:
        """注入依赖;未注入的使用全局默认。"""
        self._classifier = classifier
        self._cfg = classifier_config or IntentClassifierConfig()
        self._service = service
        self._llm = llm_provider
        self._registry = registry
        self._react_agent = react_agent
        self._plan_agent = plan_agent

    def _ensure_react_agent(self) -> object:
        if self._react_agent is None:
            service = self._service or get_vehicle_service()
            llm = self._llm or get_llm_provider()
            registry = self._registry or get_tool_registry()
            self._react_agent = build_react_agent(service, llm, registry=registry)
        return self._react_agent

    def _ensure_plan_agent(self) -> object:
        if self._plan_agent is None:
            registry = self._registry or get_tool_registry()
            self._plan_agent = build_plan_agent(registry=registry)
        return self._plan_agent

    def _apply_confidence_override(self, cls: IntentClassification) -> AgentType:
        """根据 min_confidence_for_plan 阈值修正分类结果,避免误判。"""
        if cls.agent_type == AgentType.PLAN_EXECUTE and cls.confidence < self._cfg.min_confidence_for_plan:
            logger.info(
                "Plan confidence %.2f below threshold %.2f, downgrade to REACT",
                cls.confidence, self._cfg.min_confidence_for_plan,
            )
            return AgentType.REACT
        return cls.agent_type

    def run(
        self,
        user_message: str,
        user_id: str = "demo-user",
        session_id: str = "session-001",
    ) -> OrchestratorResult:
        """执行完整意图分类→Agent 路由→结果归一化流程。"""
        classification = self._classifier.classify(user_message, user_id)
        agent_type = self._apply_confidence_override(classification)

        if agent_type == AgentType.PLAN_EXECUTE:
            agent = self._ensure_plan_agent()
            state = agent.invoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                    "user_id": user_id,
                    "session_id": session_id,
                }
            )
            response, tool_calls = _normalize_plan_output(state)
            logger.info(
                "Orchestrator dispatched PLAN agent tools=%d user=%s", len(tool_calls), user_id,
            )
        else:
            agent = self._ensure_react_agent()
            state = agent.invoke(
                {
                    "messages": [HumanMessage(content=user_message)],
                    "user_id": user_id,
                    "session_id": session_id,
                }
            )
            response, tool_calls = _normalize_react_output(state)
            logger.info(
                "Orchestrator dispatched REACT agent tools=%d user=%s", len(tool_calls), user_id,
            )

        return OrchestratorResult(
            response=response,
            tool_calls=tool_calls,
            agent_type=agent_type,
            classification=classification,
            raw_state=state,
        )


@lru_cache(maxsize=1)
def build_default_orchestrator() -> AgentOrchestrator:
    """使用全局配置构建默认编排器单例(用于 API 与 CLI)。"""
    app_cfg = get_app_config()
    classifier = build_intent_classifier(app_cfg.intent_classifier)
    return AgentOrchestrator(classifier=classifier, classifier_config=app_cfg.intent_classifier)
