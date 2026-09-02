# Memory 数据模型 - Preference/Episodic/Semantic(规格第12节)
# 运行指南:
#   from app.memory.models import Memory, MemoryType
#   被 repository/extractor/retriever/service 共用

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """记忆类型分类(规格第12节)。"""

    PREFERENCE = "preference"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Memory(BaseModel):
    """统一记忆条目,覆盖偏好/情景/语义三类。"""

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

    def is_active(self, at: datetime | None = None) -> bool:
        """判断该记忆在指定时刻是否有效(valid_to 为空或未到期)。"""
        moment = at or datetime.now(timezone.utc)
        if self.valid_to is not None and moment >= self.valid_to:
            return False
        return moment >= self.valid_from


class EpisodicEvent(BaseModel):
    """情景记忆事件(规格第12节 Episodic 示例)。"""

    event: str
    location: str | None = None
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    participants: list[str] = Field(default_factory=list)
    confidence: float = 0.7
