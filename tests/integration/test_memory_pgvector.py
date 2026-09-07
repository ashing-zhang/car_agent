# PostgreSQL + pgvector 记忆仓储集成测试(需要 Docker)
# 运行指南:
#   前置: 本机启动 Docker Desktop (Linux 容器模式)
#   运行: pytest tests/integration/test_memory_pgvector.py -v --tb=short
#   说明: 使用 testcontainers 动态拉起 pgvector/pgvector:pg16 镜像,测试结束自动销毁
#   跳过: 无 Docker 时自动 skip,不影响 CI 中无容器环境的测试

import pytest

from app.memory.embedding import MockEmbeddingProvider, reset_embedding_provider
from app.memory.models import Memory, MemoryType
from app.memory.orm import reset_db_singletons
from app.memory.repository import PostgreSQLMemoryRepository, reset_memory_repository

def _check_docker_available() -> bool:
    """检查 Docker 是否可连通(同时校验 testcontainers 可 import)。"""
    try:
        from testcontainers.postgres import PostgresContainer  # noqa: F401
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


TC_AND_DOCKER_AVAILABLE = _check_docker_available()

pytestmark = pytest.mark.skipif(
    not TC_AND_DOCKER_AVAILABLE,
    reason="testcontainers 未安装 或 Docker Desktop 未启动,跳过 pgvector 集成测试",
)

PG_VECTOR_IMAGE = "pgvector/pgvector:pg16"


@pytest.fixture()
def pg_connection_url(monkeypatch):
    """启动临时 pgvector 容器,覆盖 settings.postgres_url。

    PostgresContainer 默认返回 postgresql+psycopg2://... ;我们需要去掉 driver 前缀,
    因为项目的 create_engine 使用裸 postgres/postgresql URL,由 psycopg2 配置驱动。
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(PG_VECTOR_IMAGE, driver="psycopg2") as container:
        raw = container.get_connection_url()
        url = raw.replace("postgresql+psycopg2://", "postgresql://")
        monkeypatch.setenv("POSTGRES_URL", url)
        reset_db_singletons()
        reset_memory_repository()
        reset_embedding_provider()
        yield url
        reset_db_singletons()
        reset_memory_repository()
        reset_embedding_provider()


def _make_repo():
    return PostgreSQLMemoryRepository()


def _make_memory(user_id: str, predicate: str, value, text: str) -> Memory:
    mem = Memory(
        user_id=user_id, session_id="sess-1",
        type=MemoryType.PREFERENCE,
        predicate=predicate,
        object=str(value),
        value=value,
        confidence=0.9,
    )
    embedder = MockEmbeddingProvider()
    mem.embedding = embedder.embed(text)
    return mem


def test_create_table_and_extension(pg_connection_url) -> None:
    """首次构造 PostgreSQLMemoryRepository 会自动创建 vector 扩展和表。"""
    repo = _make_repo()
    from app.memory.orm import new_session, Base, MemoryORM
    from sqlalchemy import select, func, text

    with new_session() as session:
        ext = session.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar()
        assert ext == "vector", "pgvector 扩展未被自动创建"
        cnt = session.execute(select(func.count()).select_from(MemoryORM.__table__)).scalar()
        assert cnt == 0


def test_save_and_find_by_user(pg_connection_url) -> None:
    """save + find_by_user 基础 CRUD。"""
    repo = _make_repo()
    m1 = _make_memory("u1", "preferred_temperature", 24.0, "用户偏好车内温度24度")
    m2 = _make_memory("u1", "preferred_volume", 15, "用户偏好音量15")
    m3 = _make_memory("u2", "ac_mode", "auto", "用户偏好空调自动模式")
    repo.save(m1); repo.save(m2); repo.save(m3)

    all_u1 = repo.find_by_user("u1")
    assert {m.predicate for m in all_u1} == {"preferred_temperature", "preferred_volume"}
    all_u2 = repo.find_by_user("u2", MemoryType.PREFERENCE)
    assert len(all_u2) == 1 and all_u2[0].value == "auto"


def test_save_idempotent_upsert(pg_connection_url) -> None:
    """同 id save 两次为 Upsert(更新),不产生重复记录。"""
    repo = _make_repo()
    m1 = _make_memory("u1", "preferred_temperature", 24.0, "用户偏好24度")
    repo.save(m1)
    m1b = m1.model_copy(update={"value": 25.0, "object": "25.0celsius"})
    repo.save(m1b)
    rows = repo.find_by_user("u1")
    assert len(rows) == 1
    assert rows[0].value == 25.0


def test_find_active_filters_valid_time(pg_connection_url) -> None:
    """find_active 只返回未到期的记忆;deactivate 置为失效。"""
    repo = _make_repo()
    m1 = _make_memory("u1", "preferred_temperature", 24.0, "用户偏好温度24")
    m2 = _make_memory("u1", "preferred_temperature", 22.0, "用户偏好温度22")
    repo.save(m1)
    repo.deactivate(m1.id)
    repo.save(m2)

    active = repo.find_active("u1", "preferred_temperature")
    assert len(active) == 1
    assert active[0].id == m2.id and active[0].value == 22.0

    all_rows = repo.find_by_user("u1")
    assert len(all_rows) == 2


def test_find_similar_cosine_distance_retrieves_relevant_memory(pg_connection_url) -> None:
    """pgvector find_similar: 语义相关的记忆排前。"""
    repo = _make_repo()
    embedder = MockEmbeddingProvider()
    mems = [
        _make_memory("u1", "preferred_temperature", 24.0, "用户偏好车内温度24摄氏度"),
        _make_memory("u1", "seat_heating", 2, "用户习惯座椅加热2档"),
        _make_memory("u1", "preferred_volume", 15, "用户默认音量调到15"),
        _make_memory("u1", "home_address", "望京SOHO", "用户家地址是望京SOHO"),
    ]
    for m in mems:
        repo.save(m)

    query = "车里有点冷 你把空调温度调一下吧"
    q_vec = embedder.embed(query)
    results = repo.find_similar("u1", q_vec, top_k=3, min_similarity=0.0)
    assert len(results) >= 1
    predicates_ranked = [m.predicate for _, m in results]
    assert "preferred_temperature" in predicates_ranked


def test_find_similar_respects_user_isolation(pg_connection_url) -> None:
    """find_similar 只返回同一 user_id 的记忆(用户隔离)。"""
    repo = _make_repo()
    m_alice = _make_memory("alice", "preferred_temperature", 24.0, "Alice喜欢车内24度")
    m_bob = _make_memory("bob", "preferred_temperature", 20.0, "Bob喜欢车内20度")
    repo.save(m_alice); repo.save(m_bob)

    embedder = MockEmbeddingProvider()
    q_vec = embedder.embed("空调温度多少度")
    alice_results = repo.find_similar("alice", q_vec, top_k=5, min_similarity=0.0)
    bob_results = repo.find_similar("bob", q_vec, top_k=5, min_similarity=0.0)
    assert all(m.user_id == "alice" for _, m in alice_results)
    assert all(m.user_id == "bob" for _, m in bob_results)


def test_value_types_roundtrip_int_bool_str(pg_connection_url) -> None:
    """value 为 int / bool / str 时,从数据库读回后类型保持一致。"""
    repo = _make_repo()
    cases = [
        ("fan_speed", 3, int),
        ("seat_ventilation", "on", str),
        ("seat_massage", True, bool),
        ("home_address", "北京市朝阳区望京SOHO", str),
    ]
    for pred, val, expected_type in cases:
        mem = _make_memory("u1", pred, val, f"用户偏好{pred}={val}")
        repo.save(mem)
        back = repo.find_by_user("u1")
        loaded = [m for m in back if m.predicate == pred][0]
        assert type(loaded.value) is expected_type, f"{pred}: {type(loaded.value)} != {expected_type}"
        assert loaded.value == val
