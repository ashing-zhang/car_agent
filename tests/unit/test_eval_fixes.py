# 评估修复验证测试 - 覆盖评估失败根因的四大类修复
# 运行指南:
#   pytest tests/unit/test_eval_fixes.py -v
# 修复点:
#   1. system.md 补全能力描述(导航/媒体/多模态/车辆全能力)
#   2. 生产 LLM 调用走真实 OpenAICompatibleProvider, 单元测试用 IntentStubLLM 桩验证数据流
#   3. LLMPlanner 空计划/格式错误时降级到规则 Planner
#   4. 规则 Planner 目的地提取补充停止词(路上/途中)并改进温度提取

from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agent.llm_planner import LLMPlanner
from app.agent.planner import Planner
from app.agent.tool_registry import get_tool_registry
from app.evaluation.dataset import (
    EvalCase,
    generate_scenarios,
)
from tests.conftest import IntentStubLLM


# ---------------- 修复点 2: IntentStubLLM 桩驱动意图匹配 ----------------

def test_stub_llm_vehicle_query_calls_status() -> None:
    """vehicle_query 场景(车速/电量/续航等)应触发 get_vehicle_status。"""
    stub = IntentStubLLM()
    cases_texts = ["现在车速多少", "电量还有多少", "车辆状态如何", "当前车速", "续航里程", "车还剩多少电"]
    for text in cases_texts:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert "get_vehicle_status" in names, f"IntentStubLLM 对 '{text}' 未调用 get_vehicle_status, 实际={names}"


def test_stub_llm_vehicle_control_calls_tools() -> None:
    """vehicle_control 场景应触发 set_temperature / set_ac / get_cabin_temperature。"""
    stub = IntentStubLLM()
    cases = [
        ("把温度调到24度", "set_temperature", 24.0),
        ("把温度调到25度", "set_temperature", 25.0),
        ("开空调", "set_ac", True),
        ("关闭空调", "set_ac", False),
        ("车内温度多少", "get_cabin_temperature", None),
    ]
    for text, expected_tool, expected_val in cases:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert expected_tool in names, f"IntentStubLLM 对 '{text}' 未调用 {expected_tool}, 实际={names}"
        call = next(tc for tc in result.tool_calls if tc["name"] == expected_tool)
        if expected_tool == "set_temperature":
            assert call["args"]["temperature_c"] == expected_val
        elif expected_tool == "set_ac":
            assert call["args"]["enabled"] == expected_val


def test_stub_llm_media_calls_tools() -> None:
    """media 场景应触发 play_media / pause_media / set_volume。"""
    stub = IntentStubLLM()
    cases = [
        ("播放音乐", "play_media"),
        ("暂停播放", "pause_media"),
        ("音量调到20", "set_volume"),
        ("播放我的歌单", "play_media"),
        ("把音量设为15", "set_volume"),
        ("停止音乐", "pause_media"),
        ("放点轻音乐", "play_media"),
    ]
    for text, expected_tool in cases:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert expected_tool in names, f"IntentStubLLM 对 '{text}' 未调用 {expected_tool}, 实际={names}"


def test_stub_llm_memory_calls_set_temperature() -> None:
    """memory 偏好类场景应触发 set_temperature。"""
    stub = IntentStubLLM()
    cases = [
        ("我喜欢车内保持24度", 24.0),
        ("以后冬天保持25度", 25.0),
        ("有点冷", 24.0),
        ("太热了", 22.0),
        ("我习惯车内23度", 23.0),
        ("平时喜欢22度", 22.0),
        ("保持车内26度", 26.0),
    ]
    for text, expected_temp in cases:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对记忆场景 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert "set_temperature" in names, f"IntentStubLLM 对 '{text}' 未调用 set_temperature, 实际={names}"
        call = next(tc for tc in result.tool_calls if tc["name"] == "set_temperature")
        assert call["args"]["temperature_c"] == expected_temp, (
            f"IntentStubLLM 对 '{text}' 温度值错误, 期望={expected_temp} 实际={call['args']['temperature_c']}"
        )


def test_stub_llm_multimodal_calls_env_tools() -> None:
    """multimodal 场景应触发 get_weather / get_traffic / get_camera_scene。"""
    stub = IntentStubLLM()
    cases = [
        ("现在路况怎么样", "get_traffic"),
        ("天气如何", "get_weather"),
        ("前方交通状况", "get_traffic"),
        ("看看周围环境", "get_camera_scene"),
        ("外面下雨了吗", "get_weather"),
        ("现在还适合走原来的路线吗", "get_camera_scene"),
    ]
    for text, expected_tool in cases:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对多模态场景 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert expected_tool in names, f"IntentStubLLM 对 '{text}' 未调用 {expected_tool}, 实际={names}"


def test_stub_llm_navigation_two_step_calls() -> None:
    """导航类意图应一次性返回 search_destination + start_navigation 两步。"""
    stub = IntentStubLLM()
    cases = [
        "帮我导航到公司",
        "帮我导航到家",
        "帮我导航到医院",
        "帮我导航到机场",
        "导航回家",
        "前往机场",
        "怎么去最近的医院",
    ]
    for text in cases:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对导航 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert "search_destination" in names, f"导航 '{text}' 缺少 search_destination, 实际={names}"
        assert "start_navigation" in names, f"导航 '{text}' 缺少 start_navigation, 实际={names}"
        assert names.index("search_destination") < names.index("start_navigation"), (
            f"导航 '{text}' 两步顺序错误: {names}"
        )


def test_stub_llm_multi_step_combined() -> None:
    """多步组合场景(导航+媒体)应同时触发导航两步 + play_media。"""
    stub = IntentStubLLM()
    cases = [
        "我要去公司顺便播放音乐",
        "导航到机场并播放歌单",
        "去商场顺便放点音乐",
        "回家路上播放音乐",
        "导航去医院同时播放歌单",
        "去学校顺便放音乐",
    ]
    for text in cases:
        result = stub.invoke([HumanMessage(content=text)])
        assert result.tool_calls, f"IntentStubLLM 未对多步 '{text}' 触发工具调用"
        names = [tc["name"] for tc in result.tool_calls]
        assert "search_destination" in names, f"多步 '{text}' 缺少 search_destination, 实际={names}"
        assert "start_navigation" in names, f"多步 '{text}' 缺少 start_navigation, 实际={names}"
        assert "play_media" in names, f"多步 '{text}' 缺少 play_media, 实际={names}"


# ---------------- 修复点 4: 规则 Planner 目的地停止词与温度提取 ----------------

def test_rule_planner_destination_stop_words() -> None:
    """Planner 提取目的地时, '路上/途中/顺便' 等应作为停止词截断。"""
    planner = Planner()
    cases = [
        ("回家路上播放音乐", "家"),
        ("我要去公司顺便播放音乐", "公司"),
        ("去商场顺便放点音乐", "商场"),
        ("导航去医院同时播放歌单", "医院"),
        ("去学校途中放音乐", "学校"),
    ]
    for text, expected_dest in cases:
        dest = planner._extract_destination(text)
        assert dest == expected_dest, f"Planner 提取 '{text}' 目的地错误, 期望='{expected_dest}' 实际='{dest}'"


def test_rule_planner_memory_temperature() -> None:
    """Planner 对记忆类温度表述应正确提取目标温度。"""
    planner = Planner()
    cases = [
        ("我喜欢车内保持24度", 24.0),
        ("以后冬天保持25度", 25.0),
        ("有点冷", 24.0),
        ("太热了", 22.0),
        ("我习惯车内23度", 23.0),
        ("平时喜欢22度", 22.0),
        ("保持车内26度", 26.0),
    ]
    for text, expected_temp in cases:
        temp = planner._extract_temperature(text)
        assert temp == expected_temp, f"Planner 提取 '{text}' 温度错误, 期望={expected_temp} 实际={temp}"


def test_rule_planner_memory_cases_produce_set_temperature() -> None:
    """记忆类 eval 场景经由 Planner.plan 应产出 set_temperature 步骤。"""
    planner = Planner()
    cases = [
        "我喜欢车内保持24度",
        "以后冬天保持25度",
        "有点冷",
        "太热了",
        "我习惯车内23度",
        "平时喜欢22度",
        "保持车内26度",
    ]
    for text in cases:
        steps = planner.plan(text, "eval-user")
        tool_names = [s.tool for s in steps]
        assert "set_temperature" in tool_names, f"Planner 对 '{text}' 未产出 set_temperature, 实际={tool_names}"


def test_rule_planner_all_eval_cases_expected_tools_coverage() -> None:
    """程序化生成的全部 eval 场景中, 规则 Planner 应覆盖非空 expected_tools 的绝大多数。"""
    scenarios = generate_scenarios()
    planner = Planner()
    covered: list[EvalCase] = []
    uncovered: list[EvalCase] = []
    for case in scenarios:
        if not case.expected_tools:
            continue
        steps = planner.plan(case.user, "eval-user")
        produced = {s.tool for s in steps}
        expected = set(case.expected_tools)
        if expected.issubset(produced):
            covered.append(case)
        else:
            uncovered.append(case)
    total_checked = len(covered) + len(uncovered)
    coverage = len(covered) / total_checked if total_checked else 1.0
    assert coverage >= 0.9, (
        f"规则 Planner 对 eval 场景覆盖不足 {coverage:.0%}: 未覆盖=[{', '.join(c.id for c in uncovered[:10])}]"
    )


# ---------------- 修复点 3: LLMPlanner fallback 降级 ----------------

class _FailingLLMProvider:
    """始终返回非 JSON 内容用于触发 fallback 的假 LLM Provider。"""

    def bind_tools(self, tools: list) -> "_FailingLLMProvider":
        """遵循 LLMProvider 接口。"""
        return self

    def invoke(self, messages: list) -> object:
        """返回无法解析为 JSON 数组的纯文本。"""
        from langchain_core.messages import AIMessage
        return AIMessage(content="抱歉,我暂时无法生成计划,请稍后再试。")


def test_llm_planner_fallback_to_rule_on_invalid_output() -> None:
    """LLMPlanner 在 LLM 输出非 JSON 且重试耗尽后应降级到规则 Planner。"""
    registry = get_tool_registry()
    fake_llm = _FailingLLMProvider()
    fallback = Planner()
    planner = LLMPlanner(fake_llm, registry, rule_fallback=fallback)
    steps = planner.plan("我要去公司顺便播放音乐", "u1")
    tool_names = [s.tool for s in steps]
    assert "search_destination" in tool_names, f"fallback 后缺失 search_destination, 实际={tool_names}"
    assert "start_navigation" in tool_names, f"fallback 后缺失 start_navigation, 实际={tool_names}"
    assert "play_media" in tool_names, f"fallback 后缺失 play_media, 实际={tool_names}"


def test_llm_planner_fallback_on_empty_plan() -> None:
    """LLMPlanner 在 LLM 返回空 plan 时应降级到规则 Planner。"""

    class _EmptyLLMProvider:
        def bind_tools(self, tools: list) -> "_EmptyLLMProvider":
            return self

        def invoke(self, messages: list) -> object:
            from langchain_core.messages import AIMessage
            return AIMessage(content="[]")

    registry = get_tool_registry()
    planner = LLMPlanner(_EmptyLLMProvider(), registry, rule_fallback=Planner())
    steps = planner.plan("回家路上播放音乐", "u1")
    tool_names = [s.tool for s in steps]
    assert "search_destination" in tool_names, f"空 plan fallback 后缺失 search_destination, 实际={tool_names}"
    assert "start_navigation" in tool_names, f"空 plan fallback 后缺失 start_navigation, 实际={tool_names}"
    assert "play_media" in tool_names, f"空 plan fallback 后缺失 play_media, 实际={tool_names}"


# ---------------- 修复点 1: system prompt 完整性检查 ----------------

def test_system_prompt_mentions_all_capabilities() -> None:
    """system.md 必须显式提到导航、媒体、多模态、车辆全部能力关键词。"""
    prompt_path = Path(__file__).resolve().parent.parent.parent / "app" / "agent" / "prompts" / "system.md"
    content = prompt_path.read_text(encoding="utf-8")
    required_sections = [
        "车辆状态查询与控制",
        "导航",
        "媒体播放",
        "多模态环境感知",
        "search_destination",
        "start_navigation",
        "play_media",
        "pause_media",
        "set_volume",
        "get_weather",
        "get_traffic",
        "get_camera_scene",
        "get_vehicle_status",
        "set_temperature",
        "set_ac",
    ]
    missing = [kw for kw in required_sections if kw not in content]
    assert not missing, f"system.md 缺少关键能力描述: {missing}"


def test_evaluator_require_real_llm() -> None:
    """Evaluator 初始化时必须使用真实 LLM;缺少 API Key 时抛出 LLMConfigurationError。"""
    import pytest
    from app.config import EvalYamlConfig, EvaluatorRuntimeConfig
    from app.llm.provider import LLMConfigurationError, reset_llm_provider_singleton

    reset_llm_provider_singleton()
    from app.evaluation.evaluator import Evaluator

    eval_cfg = EvalYamlConfig(evaluator=EvaluatorRuntimeConfig(enabled=True))
    try:
        Evaluator(eval_config=eval_cfg)
    except LLMConfigurationError:
        pass
    except Exception as exc:  # noqa: BLE001 - 其他类型异常不应被静默吞噬
        pytest.fail(f"Evaluator 在无 API Key 时未抛出 LLMConfigurationError, 而是 {type(exc).__name__}: {exc}")


def test_stub_llm_evaluator_probe_case_extracts_tools() -> None:
    """对单工具场景,用 IntentStubLLM 驱动 Evaluator 能正确提取工具调用。"""
    from app.evaluation.evaluator import Evaluator
    from app.config import AppYamlConfig, EvalYamlConfig, EvaluatorRuntimeConfig
    from app.agent.graph import build_agent
    from app.simulation.providers import SceneVehicleProvider
    from app.simulation.scene_pool import get_scene_pool
    from app.tools.vehicle import VehicleService
    from app.tools.navigation import get_navigation_service
    from app.tools.media import get_media_service
    from app.tools.environment import get_environment_service
    from app.agent.tool_registry import ToolRegistry
    from app.evaluation.metrics import CaseResult, SAFETY_CRITICAL_TOOLS
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    import time

    app_cfg = AppYamlConfig()
    probe = EvalCase(
        id="probe_vq",
        user="现在车速多少",
        expected_tools=["get_vehicle_status"],
        category="vehicle_query",
    )
    service = VehicleService(SceneVehicleProvider(get_scene_pool()), app_cfg.policy)
    registry = ToolRegistry(
        service, get_navigation_service(), get_media_service(), get_environment_service()
    )
    stub = IntentStubLLM()
    agent = build_agent(service, stub, registry=registry)
    start = time.perf_counter()
    actual_tools: list[str] = []
    actual_args: list[dict] = []
    result: dict[str, list[BaseMessage]] = agent.invoke(
        {
            "messages": [HumanMessage(content=probe.user)],
            "user_id": "eval-user",
            "session_id": "probe-vq",
        }
    )
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                actual_tools.append(tc["name"])
                actual_args.append(tc["args"])
    latency = (time.perf_counter() - start) * 1000
    expected_set = set(probe.expected_tools)
    actual_set = set(actual_tools)
    success = expected_set.issubset(actual_set) if expected_set else True
    case_result = CaseResult(
        case=probe,
        actual_tools=actual_tools,
        actual_arguments=actual_args,
        success=success,
        latency_ms=latency,
    )
    assert "get_vehicle_status" in case_result.actual_tools, (
        f"Evaluator(IntentStubLLM) 未提取 get_vehicle_status, 实际={case_result.actual_tools}"
    )
    assert case_result.success is True, "probe_vq 应成功"


def test_stub_llm_evaluator_multi_step_probe() -> None:
    """对多工具场景,用规则 Planner + IntentStubLLM 驱动能正确提取全部三步。"""
    from app.evaluation.evaluator import Evaluator
    from app.config import AppYamlConfig, EvalYamlConfig, EvaluatorRuntimeConfig
    from app.agent.graph import build_agent
    from app.simulation.providers import SceneVehicleProvider
    from app.simulation.scene_pool import get_scene_pool
    from app.tools.vehicle import VehicleService
    from app.tools.navigation import get_navigation_service
    from app.tools.media import get_media_service
    from app.tools.environment import get_environment_service
    from app.agent.tool_registry import ToolRegistry
    from app.evaluation.metrics import CaseResult, SAFETY_CRITICAL_TOOLS
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    import time

    probe = EvalCase(
        id="probe_ms",
        user="回家路上播放音乐",
        expected_tools=["search_destination", "start_navigation", "play_media"],
        category="multi_step",
    )
    app_cfg = AppYamlConfig()
    service = VehicleService(SceneVehicleProvider(get_scene_pool()), app_cfg.policy)
    registry = ToolRegistry(
        service, get_navigation_service(), get_media_service(), get_environment_service()
    )
    stub = IntentStubLLM()
    agent = build_agent(service, stub, registry=registry)
    start = time.perf_counter()
    actual_tools: list[str] = []
    actual_args: list[dict] = []
    result: dict[str, list[BaseMessage]] = agent.invoke(
        {
            "messages": [HumanMessage(content=probe.user)],
            "user_id": "eval-user",
            "session_id": "probe-ms",
        }
    )
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                actual_tools.append(tc["name"])
                actual_args.append(tc["args"])
    latency = (time.perf_counter() - start) * 1000
    expected_set = set(probe.expected_tools)
    actual_set = set(actual_tools)
    success = expected_set.issubset(actual_set) if expected_set else True
    case_result = CaseResult(
        case=probe,
        actual_tools=actual_tools,
        actual_arguments=actual_args,
        success=success,
        latency_ms=latency,
    )
    missing = expected_set - actual_set
    assert not missing, (
        f"Evaluator(IntentStubLLM) 多步场景缺少工具: 缺失={missing} 实际={case_result.actual_tools}"
    )
    assert case_result.success is True, "probe_ms 应成功"
