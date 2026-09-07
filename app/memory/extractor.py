# Memory Extractor - 从对话提取记忆候选(规格第13节)
# 运行指南:
#   默认规则提取器,不依赖 LLM;后续可替换为 LLMMemoryExtractor
#   判断: durable? user-specific? useful? confidence sufficient? conflict?
#   Phase 6+: 支持座椅/媒体/导航/空调四大类偏好提取

import logging
import re

from app.memory.models import Memory, MemoryType, PREFERENCE_PREDICATES

logger = logging.getLogger(__name__)

DURABLE_KEYWORDS = ("喜欢", "偏好", "保持", "习惯", "舒服", "总是", "以后", "平常", "平时", "默认")
NAV_DURABLE_TRIGGERS = ("家在", "我家在", "家庭地址", "住址是", "公司在", "单位在", "工作地址")
EPISODIC_KEYWORDS = ("去了", "到达", "出发", "经过", "visit", "went")

AC_MODES = {"制冷": "cool", "制热": "heat", "送风": "fan", "自动": "auto", "除湿": "dry"}
ROUTE_PREFS = {
    "高速优先": "highway_first",
    "走高速": "highway_first",
    "不走高速": "no_highway",
    "躲避拥堵": "avoid_traffic",
    "距离最短": "shortest",
    "时间最短": "fastest",
}
SEAT_HEATING_LEVELS = {"低": 1, "中": 2, "高": 3, "1档": 1, "2档": 2, "3档": 3}
FAN_SPEED_LEVELS = {r"(\d+)\s*档": None}


def _extract_number(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(k in text for k in keywords)


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
        """提取偏好记忆(温度/空调/座椅/媒体/导航等)。"""
        results: list[Memory] = []
        durable = _has_any(text, DURABLE_KEYWORDS)
        nav_trigger = _has_any(text, NAV_DURABLE_TRIGGERS)
        if not (durable or nav_trigger):
            return results

        results.extend(self._extract_climate(text, user_id, session_id))
        results.extend(self._extract_seat(text, user_id, session_id))
        results.extend(self._extract_media(text, user_id, session_id))
        results.extend(self._extract_navigation(text, user_id, session_id))
        return results

    def _extract_climate(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取空调/温度类偏好。"""
        out: list[Memory] = []
        kws = PREFERENCE_PREDICATES["preferred_temperature"]
        temp_match = re.search(r"(\d+(?:\.\d+)?)\s*度", text)
        if temp_match and (_has_any(text, kws) or "度" in text):
            if temp_match:
                value = float(temp_match.group(1))
                if 10 <= value <= 40:
                    out.append(Memory(
                        user_id=user_id, session_id=session_id,
                        type=MemoryType.PREFERENCE,
                        predicate="preferred_temperature",
                        object=f"{value}celsius",
                        value=value, unit="celsius",
                        confidence=0.9, source="conversation",
                        metadata={"raw_text": text},
                    ))
        for mode_zh, mode_en in AC_MODES.items():
            if mode_zh in text and ("模式" in text or "空调" in text):
                out.append(Memory(
                    user_id=user_id, session_id=session_id,
                    type=MemoryType.PREFERENCE,
                    predicate="ac_mode",
                    object=mode_en, value=mode_en,
                    confidence=0.8, source="conversation",
                    metadata={"raw_text": text},
                ))
                break
        fan_kw = PREFERENCE_PREDICATES["fan_speed"]
        if _has_any(text, fan_kw):
            level = _extract_number(text)
            if level is not None and 1 <= level <= 7:
                out.append(Memory(
                    user_id=user_id, session_id=session_id,
                    type=MemoryType.PREFERENCE,
                    predicate="fan_speed",
                    object=f"level_{int(level)}", value=int(level),
                    confidence=0.8, source="conversation",
                    metadata={"raw_text": text},
                ))
        return out

    def _extract_seat(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取座椅类偏好(位置/加热/通风/按摩)。"""
        out: list[Memory] = []
        if _has_any(text, PREFERENCE_PREDICATES["seat_position"]):
            pos_num = _extract_number(text)
            if pos_num is not None:
                out.append(Memory(
                    user_id=user_id, session_id=session_id,
                    type=MemoryType.PREFERENCE,
                    predicate="seat_position",
                    object=f"pos_{int(pos_num)}", value=int(pos_num),
                    confidence=0.75, source="conversation",
                    metadata={"raw_text": text},
                ))
        if _has_any(text, PREFERENCE_PREDICATES["seat_heating"]):
            level_val: int | str = "on"
            for zh, num in SEAT_HEATING_LEVELS.items():
                if zh in text:
                    level_val = num
                    break
            out.append(Memory(
                user_id=user_id, session_id=session_id,
                type=MemoryType.PREFERENCE,
                predicate="seat_heating",
                object=str(level_val), value=level_val,
                confidence=0.8, source="conversation",
                metadata={"raw_text": text},
            ))
        if _has_any(text, PREFERENCE_PREDICATES["seat_ventilation"]):
            out.append(Memory(
                user_id=user_id, session_id=session_id,
                type=MemoryType.PREFERENCE,
                predicate="seat_ventilation",
                object="on", value="on",
                confidence=0.8, source="conversation",
                metadata={"raw_text": text},
            ))
        if _has_any(text, PREFERENCE_PREDICATES["seat_massage"]):
            out.append(Memory(
                user_id=user_id, session_id=session_id,
                type=MemoryType.PREFERENCE,
                predicate="seat_massage",
                object="on", value="on",
                confidence=0.75, source="conversation",
                metadata={"raw_text": text},
            ))
        return out

    def _extract_media(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取媒体类偏好(音量/音乐风格/歌单/电台)。"""
        out: list[Memory] = []
        if _has_any(text, PREFERENCE_PREDICATES["preferred_volume"]):
            vol = _extract_number(text)
            if vol is not None and 0 <= vol <= 40:
                out.append(Memory(
                    user_id=user_id, session_id=session_id,
                    type=MemoryType.PREFERENCE,
                    predicate="preferred_volume",
                    object=f"vol_{int(vol)}", value=int(vol),
                    confidence=0.85, source="conversation",
                    metadata={"raw_text": text},
                ))
        style_keywords = ("流行", "摇滚", "古典", "爵士", "民谣", "电子", "说唱", "轻音乐", "r&b", "古风")
        if _has_any(text, PREFERENCE_PREDICATES["preferred_music_style"]):
            for st in style_keywords:
                if st in text:
                    out.append(Memory(
                        user_id=user_id, session_id=session_id,
                        type=MemoryType.PREFERENCE,
                        predicate="preferred_music_style",
                        object=st, value=st,
                        confidence=0.8, source="conversation",
                        metadata={"raw_text": text},
                    ))
                    break
        if _has_any(text, PREFERENCE_PREDICATES["default_playlist"]):
            m = re.search(r"(歌单[^\s，。,.]+|喜欢的歌|我的收藏|每日推荐)", text)
            playlist = m.group(1) if m else "默认歌单"
            out.append(Memory(
                user_id=user_id, session_id=session_id,
                type=MemoryType.PREFERENCE,
                predicate="default_playlist",
                object=playlist, value=playlist,
                confidence=0.75, source="conversation",
                metadata={"raw_text": text},
            ))
        if _has_any(text, PREFERENCE_PREDICATES["preferred_radio"]):
            fm = re.search(r"fm\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
            radio_val: str | float = f"fm{fm.group(1)}" if fm else "默认电台"
            out.append(Memory(
                user_id=user_id, session_id=session_id,
                type=MemoryType.PREFERENCE,
                predicate="preferred_radio",
                object=str(radio_val), value=radio_val,
                confidence=0.7, source="conversation",
                metadata={"raw_text": text},
            ))
        return out

    def _extract_navigation(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取导航类偏好(家/公司地址、路线偏好)。"""
        out: list[Memory] = []
        address_patterns = [
            ("home_address", ("家在", "家是", "住址是", "我家在", "家庭地址")),
            ("work_address", ("公司在", "公司是", "单位在", "工作地址", "上班在")),
        ]
        for pred, patterns in address_patterns:
            for pat in patterns:
                if pat in text:
                    idx = text.index(pat) + len(pat)
                    remainder = text[idx:].strip(" ，。,.的是")
                    if remainder:
                        out.append(Memory(
                            user_id=user_id, session_id=session_id,
                            type=MemoryType.PREFERENCE,
                            predicate=pred,
                            object=remainder, value=remainder,
                            confidence=0.7, source="conversation",
                            metadata={"raw_text": text},
                        ))
                        break
        if _has_any(text, PREFERENCE_PREDICATES["route_preference"]):
            for zh_pref, en_pref in ROUTE_PREFS.items():
                if zh_pref in text:
                    out.append(Memory(
                        user_id=user_id, session_id=session_id,
                        type=MemoryType.PREFERENCE,
                        predicate="route_preference",
                        object=en_pref, value=en_pref,
                        confidence=0.8, source="conversation",
                        metadata={"raw_text": text},
                    ))
                    break
        return out

    def _extract_episodic(
        self, text: str, user_id: str, session_id: str
    ) -> list[Memory]:
        """提取情景记忆(事件)。"""
        results: list[Memory] = []
        if not _has_any(text, EPISODIC_KEYWORDS):
            return results
        event_desc = text.strip()
        results.append(Memory(
            user_id=user_id, session_id=session_id,
            type=MemoryType.EPISODIC,
            predicate="event",
            object=event_desc, value=event_desc,
            confidence=0.6, source="conversation",
            metadata={"raw_text": text},
        ))
        return results
