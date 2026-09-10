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


def test_extract_multiple_preference_categories() -> None:
    """验证 extractor 支持提取空调/座椅/媒体/导航四大类共 14 种偏好。"""
    extractor = MemoryExtractor()
    test_cases = [
        ("我喜欢车内空调保持制冷模式,风量3档", ["ac_mode", "fan_speed"]),
        ("我习惯座椅位置在3号,座椅加热开2档", ["seat_position", "seat_heating"]),
        ("我平时喜欢座椅通风和座椅按摩都打开", ["seat_ventilation", "seat_massage"]),
        ("我习惯音量设在20,喜欢听流行音乐", ["preferred_volume", "preferred_music_style"]),
        ("我默认播放歌单喜欢的歌,听广播习惯fm91.5", ["default_playlist", "preferred_radio"]),
        ("家在北京市朝阳区建国路88号", ["home_address"]),
        ("公司在海淀区中关村大街1号", ["work_address"]),
        ("我导航习惯走高速优先路线", ["route_preference"]),
    ]
    for text, expected_preds in test_cases:
        candidates = extractor.extract(text, "u_multi", "s1")
        extracted_preds = {c.predicate for c in candidates}
        for pred in expected_preds:
            assert pred in extracted_preds, f"Text '{text}' 未提取到预期偏好 {pred}, 实际: {extracted_preds}"


def test_retrieve_memory_injects_all_categories() -> None:
    """验证 retrieve_memory 节点检索并注入四大类偏好(不限于温度)。"""
    from langchain_core.messages import SystemMessage
    from app.agent.graph import _make_retrieve_memory_node
    mem_service, _ = _make_service()
    preferences_to_save = [
        "我喜欢车内温度保持24度",
        "我习惯空调自动模式",
        "我平时空调风速喜欢2档",
        "我习惯座椅加热中档",
        "我平时座椅通风保持打开",
        "我平时座椅按摩保持打开",
        "我平时音量设在18",
        "我喜欢听摇滚音乐风格",
        "我默认播放歌单我的收藏",
        "我平时听广播喜欢电台fm91.5",
        "家在朝阳区国贸",
        "公司在海淀区西二旗",
        "我导航习惯躲避拥堵路线",
    ]
    for text in preferences_to_save:
        mem_service.remember(text, "u_full", "s1")
    node_fn = _make_retrieve_memory_node(memory_service=mem_service)
    result = node_fn({"user_id": "u_full", "messages": []})
    assert "messages" in result
    assert len(result["messages"]) == 1
    sys_msg = result["messages"][0]
    assert isinstance(sys_msg, SystemMessage)
    ctx = sys_msg.content
    expected_keywords = [
        "车内温度",
        "空调模式",
        "风速",
        "座椅加热",
        "座椅通风",
        "座椅按摩",
        "音量",
        "音乐风格",
        "歌单",
        "电台",
        "家庭地址",
        "工作地址",
        "路线策略",
    ]
    for kw in expected_keywords:
        assert kw in ctx, f"上下文缺少 '{kw}' 关键字, 实际内容: {ctx}"


def test_build_preference_context_partial_prefs() -> None:
    """验证仅部分偏好存在时,只返回已存在偏好的上下文,不报错。"""
    from app.agent.graph import _build_preference_context
    mem_service, _ = _make_service()
    mem_service.remember("我平时音量设在25", "u_partial", "s1")
    ctx = _build_preference_context("u_partial", mem_service)
    assert ctx is not None
    assert "音量" in ctx
    assert "温度" not in ctx
    assert "歌单" not in ctx


def test_build_preference_context_empty_user() -> None:
    """验证无任何偏好的用户返回 None,不产生 SystemMessage。"""
    from app.agent.graph import _build_preference_context
    mem_service, _ = _make_service()
    ctx = _build_preference_context("u_empty", mem_service)
    assert ctx is None


def test_preference_config_driven_open_closed() -> None:
    """验证开闭原则:PREFERENCE_DISPLAY 配置覆盖所有 PREFERENCE_PREDICATES 谓词。"""
    from app.memory.models import PREFERENCE_PREDICATES, PREFERENCE_DISPLAY
    for predicate in PREFERENCE_PREDICATES:
        assert predicate in PREFERENCE_DISPLAY, (
            f"PREFERENCE_DISPLAY 缺少谓词 {predicate},"
            f" 新增偏好请同步扩展 PREFERENCE_DISPLAY 以支持上下文注入"
        )
        display = PREFERENCE_DISPLAY[predicate]
        assert display.label, f"{predicate} 的 label 不能为空"
        assert display.template, f"{predicate} 的 template 不能为空"
        assert "{value}" in display.template, f"{predicate} 的 template 必须包含 {{value}} 占位符"
