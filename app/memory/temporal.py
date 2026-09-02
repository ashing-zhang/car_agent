# Memory 时序规则与冲突解决(规格第12节 Temporal Rule)
# 运行指南:
#   新偏好不简单 append;旧偏好 valid_to 设为新偏好 valid_from
#   Retrieval 时 latest valid preference 优先于 historical

from datetime import datetime, timezone

from app.memory.models import Memory, MemoryType


def resolve_conflict(new_memory: Memory, existing: list[Memory]) -> list[Memory]:
    """对新偏好,将同谓词的旧有效记忆置为失效(latest_wins 策略)。"""
    if new_memory.type != MemoryType.PREFERENCE:
        return existing
    now = new_memory.valid_from
    updated: list[Memory] = []
    for mem in existing:
        if (
            mem.user_id == new_memory.user_id
            and mem.predicate == new_memory.predicate
            and mem.is_active(now)
        ):
            deactivated = mem.model_copy(update={"valid_to": now})
            updated.append(deactivated)
        else:
            updated.append(mem)
    return updated


def get_latest_active(
    memories: list[Memory], predicate: str | None = None
) -> Memory | None:
    """返回指定谓词下最新有效的记忆。"""
    now = datetime.now(timezone.utc)
    candidates = [m for m in memories if m.is_active(now)]
    if predicate is not None:
        candidates = [m for m in candidates if m.predicate == predicate]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.valid_from)


def deduplicate(memories: list[Memory]) -> list[Memory]:
    """去重:同 user/predicate/value 的只保留最新。"""
    seen: dict[tuple, Memory] = {}
    for mem in sorted(memories, key=lambda m: m.valid_from):
        key = (mem.user_id, mem.predicate, str(mem.value))
        seen[key] = mem
    return list(seen.values())
