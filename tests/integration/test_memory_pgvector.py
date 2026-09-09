# PostgreSQL + pgvector 记忆仓储集成测试(需要 Docker)
# 运行指南:
#   前置: 本机启动 Docker Desktop (Linux 容器模式)
#   运行: pytest tests/integration/test_memory_pgvector.py -v --tb=short
#   说明: 使用 testcontainers 动态拉起 pgvector/pgvector:pg16 镜像,测试结束自动销毁
#   跳过: 无 Docker 时自动 skip,不影响 CI 中无容器环境的测试

import os
from urllib.parse import quote_plus

import pytest

os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

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
PG_USER = "test"
PG_PASSWORD = "test123"
PG_DB = "car_agent_test"


def _build_postgres_url(host: str, port: int) -> str:
    """构造符合项目格式的 postgresql:// URL(使用 psycopg2)。"""
    return (
        f"postgresql://{quote_plus(PG_USER)}:{quote_plus(PG_PASSWORD)}@"
        f"{host}:{port}/{PG_DB}"
    )


@pytest.fixture()
def pg_connection_url(monkeypatch):
    """启动临时 pgvector 容器,通过环境变量覆盖 settings.postgres_url。

    直接使用 Docker SDK + 原生 postgres 环境变量方式:
    - 绕开 testcontainers 4.x PostgresContainer 的 db_name/dbname 参数冲突
    - 禁用 Ryuk sidecar 避免 Windows 下容器名/端口映射冲突
    - 轮询 TCP 端口等待 postgres 真正就绪后才 yield
    """
    import socket
    import time
    import docker

    client = docker.from_env()
    envs = {
        "POSTGRES_USER": PG_USER,
        "POSTGRES_PASSWORD": PG_PASSWORD,
        "POSTGRES_DB": PG_DB,
    }
    container = None
    try:
        container = client.containers.run(
            PG_VECTOR_IMAGE,
            environment=envs,
            detach=True,
            remove=True,
            publish_all_ports=True,
            ports={"5432/tcp": None},
            healthcheck={
                "test": ["CMD-SHELL", "pg_isready -U " + PG_USER + " -d " + PG_DB],
                "interval": 2 * 1000000000,
                "timeout": 5 * 1000000000,
                "retries": 20,
            },
        )
        deadline = time.time() + 90.0
        mapped_port = None
        host = "127.0.0.1"
        tcp_ok = False
        while time.time() < deadline and not tcp_ok:
            container.reload()
            port_info = container.attrs.get("NetworkSettings", {}).get("Ports", {}).get("5432/tcp")
            if port_info and port_info[0].get("HostPort"):
                mapped_port = int(port_info[0]["HostPort"])
                raw_host = port_info[0].get("HostIp") or ""
                if not raw_host or raw_host == "0.0.0.0" or raw_host == "::":
                    host = "127.0.0.1"
                else:
                    host = raw_host
                try:
                    with socket.create_connection((host, mapped_port), timeout=2.0):
                        tcp_ok = True
                        break
                except OSError:
                    mapped_port = None
            time.sleep(1.0)
        if not tcp_ok or mapped_port is None:
            raise RuntimeError("pgvector 容器 5432 端口未在 90s 内就绪")

        import psycopg2

        pg_ready = False
        last_err = None
        while time.time() < deadline and not pg_ready:
            try:
                conn = psycopg2.connect(
                    host=host,
                    port=mapped_port,
                    user=PG_USER,
                    password=PG_PASSWORD,
                    dbname=PG_DB,
                    connect_timeout=3,
                )
                cur = conn.cursor()
                cur.execute("SELECT 1")
                cur.close()
                conn.close()
                pg_ready = True
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(1.0)
        if not pg_ready:
            raise RuntimeError(f"PostgreSQL 在 TCP 就绪后仍未接受连接: {last_err}")

        url = _build_postgres_url(host, mapped_port)
        monkeypatch.setenv("POSTGRES_URL", url)
        from app.config import get_settings

        get_settings.cache_clear()
        reset_db_singletons()
        reset_memory_repository()
        reset_embedding_provider()
        yield url
    finally:
        errors: list[Exception] = []
        try:
            from app.config import get_settings

            get_settings.cache_clear()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        try:
            reset_db_singletons()
            reset_memory_repository()
            reset_embedding_provider()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        if container is not None:
            try:
                container.stop(timeout=5)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            try:
                container.remove(force=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        if errors:
            print(f"[pgvector-fixture cleanup warnings] {len(errors)} errors")


def _make_repo():
    return PostgreSQLMemoryRepository()


def _make_memory(user_id: str, predicate: str, value, text: str) -> Memory:
    mem = Memory(
        user_id=user_id,
        session_id="sess-1",
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
    from app.memory.orm import MemoryORM, new_session
    from sqlalchemy import func, select, text

    with new_session() as session:
        ext = session.execute(
            text("SELECT extname FROM pg_extension WHERE extname='vector'")
        ).scalar()
        assert ext == "vector", "pgvector 扩展未被自动创建"
        cnt = session.execute(
            select(func.count()).select_from(MemoryORM.__table__)
        ).scalar()
        assert cnt == 0


def test_save_and_find_by_user(pg_connection_url) -> None:
    """save + find_by_user 基础 CRUD。"""
    repo = _make_repo()
    m1 = _make_memory("u1", "preferred_temperature", 24.0, "用户偏好车内温度24度")
    m2 = _make_memory("u1", "preferred_volume", 15, "用户偏好音量15")
    m3 = _make_memory("u2", "ac_mode", "auto", "用户偏好空调自动模式")
    repo.save(m1)
    repo.save(m2)
    repo.save(m3)

    all_u1 = repo.find_by_user("u1")
    assert {m.predicate for m in all_u1} == {
        "preferred_temperature",
        "preferred_volume",
    }
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


def test_find_similar_cosine_distance_retrieves_relevant_memory(
    pg_connection_url,
) -> None:
    """pgvector find_similar: 余弦距离排序返回相似度分 + 记忆对象。

    说明: MockEmbeddingProvider 使用哈希伪随机向量,不同文本间无真实语义相似度,
    因此此处不做语义排名断言,只验证 pgvector 向量检索功能的正确性:
    1) 每条结果都包含 (similarity_float, Memory) 二元组
    2) 相似度分数按降序排列
    3) 返回结果只包含当前 user_id 且记忆的 embedding 列已正确存入
    """
    repo = _make_repo()
    embedder = MockEmbeddingProvider()
    mems = [
        _make_memory(
            "u1", "preferred_temperature", 24.0, "用户偏好车内温度24摄氏度"
        ),
        _make_memory("u1", "seat_heating", 2, "用户习惯座椅加热2档"),
        _make_memory("u1", "preferred_volume", 15, "用户默认音量调到15"),
        _make_memory("u1", "home_address", "望京SOHO", "用户家地址是望京SOHO"),
    ]
    for m in mems:
        repo.save(m)

    query = "车内温度空调冷热"
    q_vec = embedder.embed(query)
    results = repo.find_similar("u1", q_vec, top_k=3, min_similarity=-1.0)
    assert len(results) >= 1, "pgvector 余弦距离检索应返回至少一条结果"
    sims = []
    for item in results:
        assert isinstance(item, tuple) and len(item) == 2
        sim, mem = item
        assert isinstance(sim, float)
        assert isinstance(mem, Memory)
        assert mem.user_id == "u1"
        assert mem.embedding is None or len(mem.embedding) == 1536
        sims.append(sim)
    assert sims == sorted(sims, reverse=True), "结果应按相似度降序排列"


def test_find_similar_respects_user_isolation(pg_connection_url) -> None:
    """find_similar 只返回同一 user_id 的记忆(用户隔离)。"""
    repo = _make_repo()
    m_alice = _make_memory(
        "alice", "preferred_temperature", 24.0, "Alice喜欢车内24度"
    )
    m_bob = _make_memory("bob", "preferred_temperature", 20.0, "Bob喜欢车内20度")
    repo.save(m_alice)
    repo.save(m_bob)

    embedder = MockEmbeddingProvider()
    q_vec = embedder.embed("空调温度多少度")
    alice_results = repo.find_similar(
        "alice", q_vec, top_k=5, min_similarity=0.0
    )
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
        assert (
            type(loaded.value) is expected_type
        ), f"{pred}: {type(loaded.value)} != {expected_type}"
        assert loaded.value == val
