# LLM Planner - 基于真实 LLM 的多步任务规划器(委托模式,可替换规则 Planner)
# 运行指南:
#   from app.agent.llm_planner import LLMPlanner
#   planner = LLMPlanner(get_llm_provider(), get_tool_registry())
#   steps = planner.plan("我要去公司顺便播放音乐并把空调调到24度", user_id="u1")
# 若 LLM 输出格式异常,会重试 2 次,最终降级为空列表。

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.planner import Planner, get_planner
from app.agent.tool_registry import ToolRegistry
from app.llm.provider import LLMProvider
from app.tools.schemas import PlanStep

logger = logging.getLogger(__name__)

_MAX_RETRY: int = 2
_SYSTEM_PROMPT: str = """你是一名车辆智能助手的任务规划器(Planner)。根据用户的自然语言请求,生成一个有序的工具调用步骤列表。
可用工具列表(名称 - 功能说明 - 参数):
{tool_descriptions}

要求:
1. 严格按照用户意图选择必要的工具,不要遗漏也不要冗余。
2. 输出必须是合法 JSON 数组,每个元素为 {{ "id": 序号(int 从1开始), "tool": "工具名", "arguments": {{参数键值对}} }}。
3. 若没有需要调用的工具(纯闲聊/致谢等),输出空数组 []。
4. 只输出 JSON,不要输出任何 markdown 标记、解释或额外文字。
5. 注意安全:车速>120km/h 时禁止开窗,车速>5km/h 时禁止开后备箱,如用户请求不安全操作则跳过该步骤。
"""


class LLMPlanner:
    """基于 LLM 的多步任务规划器,委托 LLM 生成 PlanStep 列表,失败时降级到规则 Planner。"""

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        rule_fallback: Planner | None = None,
    ) -> None:
        """注入 LLMProvider、ToolRegistry 与可选的规则 Planner 降级器。"""
        self._llm = llm
        self._registry = registry
        self._fallback: Planner = rule_fallback or get_planner()
        self._tool_desc: str = self._build_tool_descriptions()
        logger.info("LLMPlanner initialized with %d available tools (rule fallback enabled)", len(registry.tools))

    def _build_tool_descriptions(self) -> str:
        """从 ToolRegistry 中提取工具名/说明/参数格式化为可读文本。"""
        lines: list[str] = []
        for t in self._registry.tools:
            name = getattr(t, "name", str(t))
            desc = getattr(t, "description", "")
            args_schema = getattr(t, "args_schema", None)
            params: list[str] = []
            if args_schema is not None:
                model_fields = getattr(args_schema, "model_fields", {})
                for fn, field_info in model_fields.items():
                    annotation = field_info.annotation
                    type_str = getattr(annotation, "__name__", str(annotation))
                    params.append(f"{fn}:{type_str}")
            param_str = ", ".join(params) if params else "无"
            lines.append(f"- {name} - {desc} - 参数({param_str})")
        return "\n".join(lines)

    def plan(self, user_message: str, user_id: str = "demo-user") -> list[PlanStep]:
        """委托 LLM 根据用户消息生成有序 PlanStep 列表;最多重试 2 次,失败则降级到规则 Planner。"""
        system_prompt = _SYSTEM_PROMPT.format(tool_descriptions=self._tool_desc)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        last_error: str | None = None
        content = ""
        for attempt in range(1, _MAX_RETRY + 2):
            try:
                response: AIMessage = self._llm.invoke(messages)
                content = response.content if isinstance(response.content, str) else str(response.content)
                steps = self._parse_steps(content)
                if steps:
                    logger.info(
                        "LLMPlanner generated %d steps for user=%s attempt=%d",
                        len(steps),
                        user_id,
                        attempt,
                    )
                    return steps
                logger.warning("LLMPlanner returned empty plan on attempt %d, continuing retry", attempt)
            except Exception as exc:  # noqa: BLE001 - 重试循环需要捕获任意 LLM/解析错误
                last_error = str(exc)
                logger.warning("LLMPlanner plan attempt %d failed: %s", attempt, exc)
                messages.append(AIMessage(content=content))
                messages.append(
                    HumanMessage(
                        content=f"之前的输出格式错误({exc}),请重新只输出合法 JSON 数组,不要任何额外文字。"
                    )
                )
        logger.warning(
            "LLMPlanner exhausted retries (last_error=%s), falling back to rule Planner",
            last_error,
        )
        rule_steps = self._fallback.plan(user_message, user_id)
        valid_names = set(self._registry.names)
        filtered = [s for s in rule_steps if s.tool in valid_names]
        logger.info("Rule fallback produced %d steps for user=%s", len(filtered), user_id)
        return filtered

    def _parse_steps(self, content: str) -> list[PlanStep]:
        """从 LLM 输出文本中清洗并解析 JSON 为 PlanStep 列表。"""
        cleaned = self._extract_json_array(content)
        raw: list[Any] = json.loads(cleaned)
        if not isinstance(raw, list):
            raise TypeError(f"期望 JSON 数组,得到 {type(raw).__name__}")
        valid_names = set(self._registry.names)
        steps: list[PlanStep] = []
        next_id = 1
        for item in raw:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool", ""))
            if tool_name not in valid_names:
                logger.warning("LLMPlanner skipping unknown tool: %s", tool_name)
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            step_id = int(item.get("id", next_id))
            steps.append(PlanStep(id=step_id, tool=tool_name, arguments=arguments))
            next_id = step_id + 1
        return steps

    @staticmethod
    def _extract_json_array(text: str) -> str:
        """清洗文本,提取最外层 [ ... ] JSON 数组字符串。"""
        s = text.strip()
        start = s.find("[")
        end = s.rfind("]")
        if start < 0 or end < 0 or end <= start:
            raise ValueError("LLM 输出中未找到 JSON 数组")
        return s[start : end + 1]
