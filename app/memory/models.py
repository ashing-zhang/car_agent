# Memory 数据模型 - Preference/Episodic/Semantic(规格第12节)
# 运行指南:
#   from app.memory.models import Memory, MemoryType, PREFERENCE_PREDICATES
#   被 repository/extractor/retriever/service 共用
#   Phase 6+: embedding 字段用于 pgvector 语义检索

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型分类(规格第12节)。"""

    PREFERENCE = "preference"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


PREFERENCE_PREDICATES: dict[str, tuple[str, ...]] = {
    "preferred_temperature": ("温度", "冷", "热", "暖", "空调", "ac"),
    "ac_mode": ("空调模式", "制冷", "制热", "送风", "自动", "除湿"),
    "fan_speed": ("风速", "风量"),
    "seat_position": ("座椅位置", "座椅前后", "座椅高低"),
    "seat_heating": ("座椅加热", "加热"),
    "seat_ventilation": ("座椅通风", "通风"),
    "seat_massage": ("座椅按摩", "按摩"),
    "preferred_volume": ("音量", "声音大小"),
    "preferred_music_style": ("音乐风格", "曲风", "喜欢听", "音乐类型"),
    "default_playlist": ("歌单", "播放列表", "默认歌单"),
    "preferred_radio": ("电台", "广播", "fm"),
    "home_address": ("家", "回家", "家庭地址", "住址"),
    "work_address": ("公司", "上班", "工作地址", "单位"),
    "route_preference": ("路线偏好", "走高速", "不走高速", "躲避拥堵"),
}


class PreferenceDisplay(BaseModel):
    """偏好显示配置:中文标签与上下文模板(开闭原则:新增偏好仅需扩展此处与 PREFERENCE_PREDICATES)。"""

    label: str
    template: str


PREFERENCE_DISPLAY: dict[str, PreferenceDisplay] = {
    "preferred_temperature": PreferenceDisplay(
        label="车内温度偏好",
        template="已知用户偏好:车内温度 {value}{unit}。若用户表达冷热感受,优先采用该偏好温度设置空调。",
    ),
    "ac_mode": PreferenceDisplay(
        label="空调模式偏好",
        template="已知用户偏好:空调模式 {value}。若用户需要调节空调,优先使用此模式(制冷/制热/送风/自动/除湿)。",
    ),
    "fan_speed": PreferenceDisplay(
        label="风速偏好",
        template="已知用户偏好:空调风速 {value} 档。若用户开启或调节空调,优先使用该风速档位。",
    ),
    "seat_position": PreferenceDisplay(
        label="座椅位置偏好",
        template="已知用户偏好:座椅位置 {value}。若用户上车或调整座椅,可建议使用此预设位置。",
    ),
    "seat_heating": PreferenceDisplay(
        label="座椅加热偏好",
        template="已知用户偏好:座椅加热 {value}。若天气寒冷或用户提及座椅冷,优先开启对应档位的座椅加热。",
    ),
    "seat_ventilation": PreferenceDisplay(
        label="座椅通风偏好",
        template="已知用户偏好:座椅通风 {value}。若天气炎热或用户提及座椅闷热,优先开启座椅通风。",
    ),
    "seat_massage": PreferenceDisplay(
        label="座椅按摩偏好",
        template="已知用户偏好:座椅按摩 {value}。若用户长途驾驶或提及疲劳,可建议开启座椅按摩。",
    ),
    "preferred_volume": PreferenceDisplay(
        label="音量偏好",
        template="已知用户偏好:媒体音量 {value}。若用户播放媒体,优先设置为该音量。",
    ),
    "preferred_music_style": PreferenceDisplay(
        label="音乐风格偏好",
        template="已知用户偏好:音乐风格 {value}。若用户请求播放音乐,优先推荐该风格的歌曲或歌单。",
    ),
    "default_playlist": PreferenceDisplay(
        label="默认歌单偏好",
        template="已知用户偏好:默认歌单「{value}」。若用户未指定播放内容,可播放此歌单。",
    ),
    "preferred_radio": PreferenceDisplay(
        label="电台偏好",
        template="已知用户偏好:电台 {value}。若用户想听广播,优先推荐此电台。",
    ),
    "home_address": PreferenceDisplay(
        label="家庭地址偏好",
        template="已知用户偏好:家庭地址「{value}」。若用户提及回家,可直接导航到该地址。",
    ),
    "work_address": PreferenceDisplay(
        label="工作地址偏好",
        template="已知用户偏好:工作地址「{value}」。若用户提及上班/去公司,可直接导航到该地址。",
    ),
    "route_preference": PreferenceDisplay(
        label="路线偏好",
        template="已知用户偏好:路线策略 {value}。若用户发起导航,优先采用该路线偏好(高速优先/躲避拥堵/不走高速等)。",
    ),
}


def memory_to_text(memory: "Memory") -> str:
    """将记忆条目序列化为自然语言文本,用于 embedding。"""
    type_label = {
        MemoryType.PREFERENCE: "用户偏好",
        MemoryType.EPISODIC: "情景事件",
        MemoryType.SEMANTIC: "语义知识",
    }.get(memory.type, "记忆")
    unit = memory.unit or ""
    return f"{type_label}: {memory.predicate} = {memory.value}{unit} ({memory.object})"


class Memory(BaseModel):
    """统一记忆条目,覆盖偏好/情景/语义三类。

    embedding 字段可选,保存时由 Service 层按需计算并填充。
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str
    session_id: str
    type: MemoryType
    subject: str = "user"
    predicate: str
    object: str
    value: float | int | str | bool
    unit: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    source: str = "conversation"
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: datetime | None = None
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None)

    def is_active(self, at: datetime | None = None) -> bool:
        """判断该记忆在指定时刻是否有效(valid_to 为空或未到期)。"""
        moment = at or datetime.now(timezone.utc)
        if self.valid_to is not None and moment >= self.valid_to:
            return False
        return moment >= self.valid_from

    def to_text(self) -> str:
        """返回该记忆的自然语言表示,用于 embedding 计算。"""
        return memory_to_text(self)


class EpisodicEvent(BaseModel):
    """情景记忆事件(规格第12节 Episodic 示例)。"""

    event: str
    location: str | None = None
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    participants: list[str] = Field(default_factory=list)
    confidence: float = 0.7
