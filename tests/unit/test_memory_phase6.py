# Phase 6+ Memory 扩展测试 - 多偏好提取 + 向量语义检索
# 运行指南: pytest tests/unit/test_memory_phase6.py -v

from app.memory.embedding import (
    MockEmbeddingProvider,
    get_embedding_provider,
    reset_embedding_provider,
)
from app.memory.extractor import MemoryExtractor
from app.memory.models import Memory, MemoryType, PREFERENCE_PREDICATES
from app.memory.repository import (
    InMemoryMemoryRepository,
    build_repository,
    reset_memory_repository,
)
from app.memory.retriever import MemoryRetriever
from app.memory.service import MemoryService, reset_memory_service


def _make_svc() -> tuple[MemoryService, InMemoryMemoryRepository, MockEmbeddingProvider]:
    reset_memory_service()
    reset_memory_repository()
    reset_embedding_provider()
    embed = MockEmbeddingProvider()
    repo = build_repository("in_memory")
    assert isinstance(repo, InMemoryMemoryRepository)
    retriever = MemoryRetriever(repo, top_k=5, embedding_provider=embed)
    extractor = MemoryExtractor()
    svc = MemoryService(extractor, retriever, repo, embedding_provider=embed)
    return svc, repo, embed


def test_embedding_provider_deterministic() -> None:
    """相同文本 -> 相同向量;不同文本 -> 不同向量。"""
    p = MockEmbeddingProvider()
    v1 = p.embed("用户偏好温度24度")
    v2 = p.embed("用户偏好温度24度")
    v3 = p.embed("去公司导航")
    assert len(v1) == 1536
    assert v1 == v2
    assert v1 != v3
    assert abs(sum(x * x for x in v1) - 1.0) < 0.01


def test_extract_seat_preferences() -> None:
    """提取座椅加热/通风/按摩偏好。"""
    ext = MemoryExtractor(min_confidence=0.0)
    cases = [
        ("我习惯开2档座椅加热", "seat_heating", 2),
        ("我喜欢座椅通风保持开启", "seat_ventilation", "on"),
        ("以后开车总是给我开座椅按摩", "seat_massage", "on"),
    ]
    for text, pred, expected in cases:
        cands = ext.extract(text, "u", "s")
        matched = [c for c in cands if c.predicate == pred]
        assert matched, f"{text} 应该提取 {pred}"
        assert matched[0].value == expected, f"{text}: {matched[0].value} != {expected}"


def test_extract_media_preferences() -> None:
    """提取媒体偏好(音量/歌单/音乐风格)。"""
    ext = MemoryExtractor(min_confidence=0.0)
    cands = ext.extract("我平时喜欢把音量默认调到15", "u", "s")
    vols = [c for c in cands if c.predicate == "preferred_volume"]
    assert vols and vols[0].value == 15

    cands = ext.extract("我平常喜欢听流行风格音乐", "u", "s")
    styles = [c for c in cands if c.predicate == "preferred_music_style"]
    assert styles and styles[0].value == "流行"

    cands = ext.extract("以后默认播放我的收藏歌单", "u", "s")
    pls = [c for c in cands if c.predicate == "default_playlist"]
    assert pls


def test_extract_navigation_preferences() -> None:
    """提取导航偏好(家/公司地址、路线偏好)。"""
    ext = MemoryExtractor(min_confidence=0.0)
    cands = ext.extract("我家在北京市朝阳区望京SOHO", "u", "s")
    homes = [c for c in cands if c.predicate == "home_address"]
    assert homes and "望京" in str(homes[0].value)

    cands = ext.extract("上班的话我习惯走高速优先", "u", "s")
    routes = [c for c in cands if c.predicate == "route_preference"]
    assert routes and routes[0].value == "highway_first"


def test_extract_ac_mode_and_fan() -> None:
    """提取空调模式和风速偏好。"""
    ext = MemoryExtractor(min_confidence=0.0)
    cands = ext.extract("我喜欢空调自动模式", "u", "s")
    modes = [c for c in cands if c.predicate == "ac_mode"]
    assert modes and modes[0].value == "auto"

    cands = ext.extract("以后空调风速默认调到3档", "u", "s")
    fans = [c for c in cands if c.predicate == "fan_speed"]
    assert fans and fans[0].value == 3


def test_memory_embedding_saved_on_remember() -> None:
    """保存记忆时自动填充 embedding。"""
    svc, _, _ = _make_svc()
    saved = svc.remember("我喜欢车内保持24度", "u1", "s1")
    assert saved
    assert saved[0].embedding is not None
    assert len(saved[0].embedding) == 1536


def test_semantic_vector_retrieval_in_memory() -> None:
    """InMemory 后端基于 embedding 的余弦相似度检索。"""
    svc, _, embed = _make_svc()
    svc.remember("我喜欢车内保持24度", "u1", "s1")
    svc.remember("我习惯座椅加热开2档", "u1", "s1")
    svc.remember("平时我喜欢把音量调到15", "u1", "s1")
    svc.remember("以后默认播放流行歌单", "u1", "s1")
    svc.remember("我家在望京SOHO", "u1", "s1")

    query = "车里有点冷,你调一下温度"
    results = svc.recall_relevant("u1", query)
    assert len(results) >= 1
    predicates = [r.predicate for r in results]
    assert "preferred_temperature" in predicates


def test_keyword_fallback_works() -> None:
    """无 embedding 时仍可通过关键词检索。"""
    embed = MockEmbeddingProvider()
    repo = InMemoryMemoryRepository()
    mem = Memory(
        user_id="u1", session_id="s1",
        type=MemoryType.PREFERENCE,
        predicate="preferred_temperature",
        object="24celsius", value=24.0, unit="celsius",
        embedding=None,
    )
    repo.save(mem)
    retriever = MemoryRetriever(repo, top_k=3, embedding_provider=embed)
    results = retriever.retrieve_relevant("u1", "现在多少度空调温度")
    assert len(results) >= 1
    assert results[0].predicate == "preferred_temperature"


def test_predicate_constants_complete() -> None:
    """四大类偏好 predicate 全部已在常量中定义。"""
    expected = [
        "preferred_temperature", "ac_mode", "fan_speed",
        "seat_position", "seat_heating", "seat_ventilation", "seat_massage",
        "preferred_volume", "preferred_music_style", "default_playlist", "preferred_radio",
        "home_address", "work_address", "route_preference",
    ]
    for pred in expected:
        assert pred in PREFERENCE_PREDICATES, f"{pred} not in PREFERENCE_PREDICATES"
        assert isinstance(PREFERENCE_PREDICATES[pred], tuple)
        assert len(PREFERENCE_PREDICATES[pred]) > 0
