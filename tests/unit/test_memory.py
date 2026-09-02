# Memory 模块单元测试 - 使用独立 service 实例,不依赖全局单例与外部服务
# 运行指南: pytest tests/unit/test_memory.py -v

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_agent
from app.config import PolicyConfig
from app.llm.provider import MockLLMProvider
from app.memory.extractor import MemoryExtractor
from app.memory.models import Memory, MemoryType
from app.memory.repository import InMemoryMemoryRepository
from app.memory.retriever import MemoryRetriever
from app.memory.service import MemoryService
from app.memory.temporal import get_latest_active, resolve_conflict
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.tools.vehicle import VehicleService


def _make_service() -> tuple[MemoryService, InMemoryMemoryRepository]:
    """构造独立的 MemoryService 与仓储。"""
    repo = InMemoryMemoryRepository()
    retriever = MemoryRetriever(repo)
    extractor = MemoryExtractor()
    return MemoryService(extractor, retriever, repo), repo


def test_extract_durable_preference() -> None:
    """验证"我喜欢车内保持24度"提取出偏好温度 24。"""
    extractor = MemoryExtractor()
    candidates = extractor.extract("我喜欢车内保持24度", "u1", "s1")
    assert len(candidates) == 1
    assert candidates[0].value == 24.0
    assert candidates[0].type == MemoryType.PREFERENCE


def test_extract_not_durable_cold_today() -> None:
    """验证"今天有点冷"不提取为持久记忆(规格第13节)。"""
    extractor = MemoryExtractor()
    candidates = extractor.extract("今天有点冷", "u1", "s1")
    assert len(candidates) == 0


def test_save_and_recall_preference() -> None:
    """验证保存偏好后可检索到。"""
    service, _ = _make_service()
    service.remember("我喜欢车内保持24度", "u1", "s1")
    pref = service.recall_preference("u1", "preferred_temperature")
    assert pref is not None
    assert pref.value == 24.0


def test_conflict_resolution_latest_wins() -> None:
    """验证新偏好 23 覆盖旧偏好 25,旧偏好失效(规格第12节时序规则)。"""
    service, repo = _make_service()
    service.remember("我喜欢车内保持25度", "u1", "s1")
    service.remember("我喜欢车内保持23度", "u1", "s2")
    pref = service.recall_preference("u1", "preferred_temperature")
    assert pref is not None
    assert pref.value == 23.0
    active = repo.find_active("u1", "preferred_temperature")
    assert len(active) == 1


def test_temporal_resolve_conflict_deactivates_old() -> None:
    """验证 resolve_conflict 将旧记忆 valid_to 置为非空。"""
    old = Memory(
        user_id="u1", session_id="s1", type=MemoryType.PREFERENCE,
        predicate="preferred_temperature", object="25", value=25.0, unit="celsius",
    )
    new = Memory(
        user_id="u1", session_id="s2", type=MemoryType.PREFERENCE,
        predicate="preferred_temperature", object="23", value=23.0, unit="celsius",
    )
    resolved = resolve_conflict(new, [old])
    assert resolved[0].valid_to is not None
    latest = get_latest_active(resolved + [new], "preferred_temperature")
    assert latest is not None
    assert latest.value == 23.0


def test_personalized_agent_uses_memory() -> None:
    """验证跨会话个性化:第一次存偏好24,第二次"有点冷"用24(规格 Demo C)。"""
    mem_service, _ = _make_service()
    service = VehicleService(SceneVehicleProvider(get_scene_pool()), PolicyConfig())
    llm = MockLLMProvider()
    agent = build_agent(service, llm, memory_service=mem_service)

    agent.invoke({"messages": [HumanMessage(content="我喜欢车内保持24度")], "user_id": "u1", "session_id": "s1"})
    result = agent.invoke({"messages": [HumanMessage(content="有点冷")], "user_id": "u1", "session_id": "s2"})

    found_temp: float | None = None
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "set_temperature":
                    found_temp = tc["args"]["temperature_c"]
    assert found_temp == 24.0


def test_no_preference_falls_back_to_default() -> None:
    """验证无偏好时"有点冷"回退默认24。"""
    mem_service, _ = _make_service()
    service = VehicleService(SceneVehicleProvider(get_scene_pool()), PolicyConfig())
    llm = MockLLMProvider()
    agent = build_agent(service, llm, memory_service=mem_service)
    result = agent.invoke({"messages": [HumanMessage(content="有点冷")], "user_id": "u2", "session_id": "s1"})
    found_temp: float | None = None
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "set_temperature":
                    found_temp = tc["args"]["temperature_c"]
    assert found_temp == 24.0
