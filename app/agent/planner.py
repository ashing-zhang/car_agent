# Planner - 多步任务规划,输出结构化 Plan(规格第16节)
# 运行指南:
#   planner = get_planner()
#   steps = planner.plan("我要去公司顺便播放音乐", user_id="u1")
# 禁止让 LLM 输出任意 Python 代码执行;Plan 仅包含 tool + arguments

import logging
import re

from app.tools.schemas import PlanStep

logger = logging.getLogger(__name__)

DEST_KEYWORDS = ("去", "导航到", "出发去", "回", "前往")
MEDIA_KEYWORDS = ("音乐", "播放", "歌单", "听")
TEMP_KEYWORDS = ("温度", "冷", "热", "调到", "调成")


class Planner:
    """基于规则的多步任务规划器,生成有序 PlanStep 列表。"""

    def plan(self, user_message: str, user_id: str = "demo-user") -> list[PlanStep]:
        """根据用户消息生成有序执行步骤,保证依赖顺序。"""
        steps: list[PlanStep] = []
        step_id = 1

        dest = self._extract_destination(user_message)
        if dest is not None:
            steps.append(PlanStep(id=step_id, tool="search_destination", arguments={"query": dest}))
            step_id += 1
            steps.append(PlanStep(id=step_id, tool="start_navigation", arguments={"destination": f"{dest} 大厦"}))
            step_id += 1

        playlist = self._extract_playlist(user_message)
        if playlist is not None:
            steps.append(PlanStep(id=step_id, tool="play_media", arguments={"playlist": playlist}))
            step_id += 1

        temp = self._extract_temperature(user_message)
        if temp is not None:
            steps.append(PlanStep(id=step_id, tool="set_temperature", arguments={"temperature_c": temp}))
            step_id += 1

        volume = self._extract_volume(user_message)
        if volume is not None:
            steps.append(PlanStep(id=step_id, tool="set_volume", arguments={"level": volume}))
            step_id += 1

        logger.info("Planner generated %d steps for: %s", len(steps), user_message)
        return steps

    def _extract_destination(self, text: str) -> str | None:
        """从消息中提取目的地。"""
        for kw in DEST_KEYWORDS:
            idx = text.find(kw)
            if idx >= 0:
                rest = text[idx + len(kw):]
                for stop in ("顺便", "然后", "同时", "，", ",", "。", "."):
                    pos = rest.find(stop)
                    if pos >= 0:
                        rest = rest[:pos]
                rest = rest.strip()
                if rest:
                    return rest
        return None

    def _extract_playlist(self, text: str) -> str | None:
        """从消息中提取播放列表名。"""
        if not any(k in text for k in MEDIA_KEYWORDS):
            return None
        if "上班" in text or "工作" in text or "平时" in text:
            return "上班通勤歌单"
        if "回家" in text or "下班" in text:
            return "下班放松歌单"
        return "默认歌单"

    def _extract_temperature(self, text: str) -> float | None:
        """从消息中提取目标温度。"""
        if not any(k in text for k in TEMP_KEYWORDS):
            return None
        match = re.search(r"(\d+(?:\.\d+)?)\s*度?", text)
        if match:
            return float(match.group(1))
        if "冷" in text:
            return 24.0
        if "热" in text:
            return 22.0
        return None

    def _extract_volume(self, text: str) -> int | None:
        """从消息中提取音量。"""
        if "音量" not in text:
            return None
        match = re.search(r"(\d+)", text)
        if match:
            return int(match.group(1))
        return None


_default_planner: Planner | None = None


def get_planner() -> Planner:
    """获取默认 Planner 单例。"""
    global _default_planner
    if _default_planner is None:
        _default_planner = Planner()
    return _default_planner
