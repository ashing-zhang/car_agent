# Memory ORM Layer - SQLAlchemy + pgvector 数据模型
# 运行指南:
#   需先启动 pgvector 容器: docker compose up -d postgres
#   首次初始化会自动创建扩展和表;向量维度 1536 与 embedding 层对齐

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings
from app.memory.embedding import DEFAULT_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class MemoryORM(Base):
    """Memory 表 ORM 映射,使用 pgvector Vector 列存储嵌入。"""

    __tablename__ = "memories"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    type = Column(String(32), nullable=False, index=True)
    subject = Column(String(64), default="user")
    predicate = Column(String(128), nullable=False, index=True)
    object = Column(Text, nullable=False)
    value_num = Column(Float, nullable=True)
    value_str = Column(Text, nullable=True)
    value_bool = Column(Boolean, nullable=True)
    unit = Column(String(32), nullable=True)
    confidence = Column(Float, default=0.5)
    source = Column(String(64), default="conversation")
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_to = Column(DateTime(timezone=True), nullable=True)
    metadata_json = Column(JSON, default=dict)
    embedding = Column(Vector(DEFAULT_EMBEDDING_DIM), nullable=True)

    __table_args__ = (
        Index("ix_memories_user_predicate", "user_id", "predicate"),
        Index("ix_memories_user_type_valid", "user_id", "type", "valid_from"),
    )


def _serialize_value(value: Any) -> dict:
    """将 Pydantic Memory.value 拆分到 ORM 的三个 value 列。"""
    out = {"value_num": None, "value_str": None, "value_bool": None}
    if isinstance(value, bool):
        out["value_bool"] = value
        out["value_str"] = str(value)
    elif isinstance(value, (int, float)):
        out["value_num"] = float(value)
        out["value_str"] = str(value)
    else:
        out["value_str"] = str(value)
    return out


def _deserialize_value(orm: MemoryORM) -> Any:
    """从 ORM 三列还原 Pydantic Memory.value。"""
    if orm.value_bool is not None and orm.value_str in ("True", "False", "true", "false"):
        return orm.value_bool
    if orm.value_num is not None:
        if orm.value_str is not None and "." not in orm.value_str:
            try:
                return int(orm.value_str)
            except ValueError:
                pass
        return orm.value_num
    return orm.value_str


def memory_to_orm(mem) -> MemoryORM:
    val = _serialize_value(mem.value)
    return MemoryORM(
        id=mem.id,
        user_id=mem.user_id,
        session_id=mem.session_id,
        type=mem.type.value if hasattr(mem.type, "value") else str(mem.type),
        subject=mem.subject,
        predicate=mem.predicate,
        object=mem.object,
        value_num=val["value_num"],
        value_str=val["value_str"],
        value_bool=val["value_bool"],
        unit=mem.unit,
        confidence=mem.confidence,
        source=mem.source,
        valid_from=mem.valid_from,
        valid_to=mem.valid_to,
        metadata_json=mem.metadata or {},
        embedding=mem.embedding if mem.embedding else None,
    )


def orm_to_memory(orm: MemoryORM):
    from app.memory.models import Memory, MemoryType

    type_val = MemoryType(orm.type) if orm.type in {e.value for e in MemoryType} else orm.type
    return Memory(
        id=orm.id,
        user_id=orm.user_id,
        session_id=orm.session_id,
        type=type_val,
        subject=orm.subject,
        predicate=orm.predicate,
        object=orm.object,
        value=_deserialize_value(orm),
        unit=orm.unit,
        confidence=orm.confidence,
        source=orm.source,
        valid_from=orm.valid_from,
        valid_to=orm.valid_to,
        metadata=orm.metadata_json if orm.metadata_json else {},
        embedding=list(orm.embedding) if orm.embedding is not None else None,
    )


_default_engine = None
_default_session_factory = None


def get_engine():
    """获取 SQLAlchemy engine 单例。"""
    global _default_engine
    if _default_engine is None:
        settings = get_settings()
        _default_engine = create_engine(settings.postgres_url, pool_pre_ping=True, future=True)
    return _default_engine


def get_session_factory():
    """获取 sessionmaker 单例,并确保表与 pgvector 扩展已创建。"""
    global _default_session_factory
    if _default_session_factory is None:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)
        _default_session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return _default_session_factory


def new_session() -> Session:
    """创建新的数据库 Session。"""
    return get_session_factory()()


def reset_db_singletons() -> None:
    """重置 engine/session 单例(测试用)。"""
    global _default_engine, _default_session_factory
    _default_engine = None
    _default_session_factory = None
