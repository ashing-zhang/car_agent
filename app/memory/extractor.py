# Memory Extractor - 从对话提取记忆候选(规格第13节)
# 运行指南:
#   默认规则提取器,不依赖 LLM;后续可替换为 LLMMemoryExtractor
#   判断: durable? user-specific? useful? confidence sufficient? conflict?

import logging
import re

from app.memory.models import Memory, MemoryType

logger = logging.getLogger(__name__)

DURABLE_KEYWORDS = ("喜欢", "偏好", "保持", "习惯", "舒服", "总是", "以后", "平常", "平时")
EPISODIC_KEYWORDS = ("去了", "到达", "出发", "经过", "visit", "went")


class MemoryExtractor:
    """基于规则的记忆候选提取器。"""

    def __init__(self, min_confidence: float = 0.5) -> None:
        """设置最低置信度阈值。"""
        self._min_confidence = min_confidence

    def extract(
        self, user_message: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """从用户消息提取候选记忆列表。"""
        candidates: list[Memory] = []
        candidates.extend(self._extract_preferences(user_message, user_id, session_id))
        candidates.extend(self._extract_episodic(user_message, user_id, session_id))
        kept = [c for c in candidates if c.confidence >= self._min_confidence]
        logger.info("Extracted %d memory candidates (kept %d)", len(candidates), len(kept))
        return kept

    def _extract_preferences(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取偏好记忆(温度等)。"""
        results: list[Memory] = []
        if not any(k in text for k in DURABLE_KEYWORDS):
            return results
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*度?", text)
        if temp_match and any(k in text for k in ("温度", "冷", "热", "暖")):
            value = float(temp_match.group(1))
            results.append(
                Memory(
                    user_id=user_id,
                    session_id=session_id,
                    type=MemoryType.PREFERENCE,
                    predicate="preferred_temperature",
                    object=f"{value}celsius",
                    value=value,
                    unit="celsius",
                    confidence=0.9,
                    source="conversation",
                    metadata={"raw_text": text},
                )
            )
        return results

    def _extract_episodic(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取情景记忆(事件)。"""
        results: list[Memory] = []
        if not any(k in text for k in EPISODIC_KEYWORDS):
            return results
        event_desc = text.strip()
        results.append(
            Memory(
                user_id=user_id,
                session_id=session_id,
                type=MemoryType.EPISODIC,
                predicate="event",
                object=event_desc,
                value=event_desc,
                confidence=0.6,
                source="conversation",
                metadata={"raw_text": text},
            )
        )
        return results
