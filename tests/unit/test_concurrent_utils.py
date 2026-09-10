# 高并发测试辅助模块的单元测试
# 运行指南:
#   单独运行: pytest tests/unit/test_concurrent_utils.py -v
#   注意: 此文件测试 concurrent_utils.py 中的纯工具函数,不依赖 FastAPI app

import math

from tests.unit.concurrent_utils import (
    ConcurrencyStats,
    RequestResult,
    ThresholdsConfig,
    _percentile,
    assert_thresholds,
    compute_statistics,
    generate_user_ids,
    now_ms,
    pick_random_message,
    pick_random_temperature,
    random_jitter,
)


def test_generate_user_ids_unique_and_format() -> None:
    """验证生成的用户 ID 数量正确、格式正确且不重复。"""
    count = 10
    ids = generate_user_ids(count)
    assert len(ids) == count
    assert len(set(ids)) == count
    assert ids[0] == "concurrent-user-0000"
    assert ids[-1] == "concurrent-user-0009"


def test_random_jitter_in_range() -> None:
    """验证随机延迟在指定范围内。"""
    for _ in range(50):
        j = random_jitter(0.1, 0.5)
        assert 0.1 <= j <= 0.5
    same = random_jitter(0.3, 0.3)
    assert same == 0.3
    zero = random_jitter(0.0, 0.0)
    assert zero == 0.0


def test_pick_random_message_and_temp() -> None:
    """验证随机消息和温度选择功能(含空输入的默认值)。"""
    default_msg = pick_random_message(None)
    assert isinstance(default_msg, str)
    assert default_msg == "你好"
    default_temp = pick_random_temperature(None)
    assert default_temp == 24.0

    msgs = ["a", "b", "c"]
    for _ in range(20):
        assert pick_random_message(msgs) in msgs
    temps = [18.0, 22.0, 26.0, 30.0]
    for _ in range(20):
        assert pick_random_temperature(temps) in temps


def test_now_ms_monotonic() -> None:
    """验证时间戳函数返回正数且单调递增。"""
    t1 = now_ms()
    t2 = now_ms()
    assert t1 > 0
    assert t2 >= t1


def test_percentile_single_and_empty() -> None:
    """验证百分位数计算的边界情况。"""
    assert _percentile([], 50) == 0.0
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 99) == 42.0


def test_percentile_even_and_odd() -> None:
    """验证偶数/奇数长度列表的百分位数计算(线性插值)。"""
    data_5 = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(_percentile(data_5, 50), 3.0, abs_tol=1e-9)
    assert math.isclose(_percentile(data_5, 0), 1.0, abs_tol=1e-9)
    assert math.isclose(_percentile(data_5, 100), 5.0, abs_tol=1e-9)
    data_4 = sorted([10.0, 20.0, 30.0, 40.0])
    assert math.isclose(_percentile(data_4, 50), 25.0, abs_tol=1e-9)


def _make_mock_results(total: int, fail_rate: float = 0.0, base_latency: float = 100.0) -> list[RequestResult]:
    """构造模拟请求结果用于统计测试。"""
    results: list[RequestResult] = []
    for i in range(total):
        fail = i < int(total * fail_rate)
        results.append(RequestResult(
            user_id=f"u{i}",
            session_id=f"s{i}",
            request_index=0,
            success=not fail,
            status_code=200 if not fail else 500,
            latency_ms=base_latency + i * 10.0,
            error_message="boom" if fail else None,
            endpoint="/test",
        ))
    return results


def test_compute_statistics_empty() -> None:
    """验证空结果列表的统计值。"""
    stats = compute_statistics("/empty", [], 1000.0)
    assert stats.total_requests == 0
    assert stats.success_rate == 0.0
    assert stats.endpoint == "/empty"


def test_compute_statistics_all_success() -> None:
    """验证全成功场景的统计指标正确性。"""
    results = _make_mock_results(10, fail_rate=0.0, base_latency=100.0)
    stats = compute_statistics("/ok", results, 5000.0)
    assert stats.total_requests == 10
    assert stats.successful_requests == 10
    assert stats.failed_requests == 0
    assert stats.success_rate == 1.0
    assert stats.latency_min_ms == 100.0
    assert stats.latency_max_ms == 190.0
    assert stats.latency_mean_ms == 145.0
    assert len(stats.errors) == 0
    assert math.isclose(stats.requests_per_second, (10 * 1000.0) / 5000.0)


def test_compute_statistics_with_failures() -> None:
    """验证含失败请求的统计指标正确性。"""
    results = _make_mock_results(20, fail_rate=0.25, base_latency=50.0)
    stats = compute_statistics("/partial", results, 8000.0)
    assert stats.total_requests == 20
    assert stats.failed_requests == 5
    assert stats.successful_requests == 15
    assert stats.success_rate == 0.75
    assert len(stats.errors) == 5
    summary = stats.summary()
    assert "/partial" in summary
    assert "成功率=75.00%" in summary or "成功率=75%" in summary or "0.75" in summary


def test_assert_thresholds_pass() -> None:
    """验证阈值断言在指标达标时不抛异常。"""
    results = _make_mock_results(100, fail_rate=0.0, base_latency=10.0)
    stats = compute_statistics("/good", results, 2000.0)
    thresholds = ThresholdsConfig(
        min_success_rate=0.95,
        max_p50_latency_ms=200.0,
        max_p95_latency_ms=500.0,
        max_p99_latency_ms=1000.0,
    )
    assert_thresholds(stats, thresholds)


def test_assert_thresholds_fail_success_rate() -> None:
    """验证成功率不达标时断言失败。"""
    results = _make_mock_results(100, fail_rate=0.10, base_latency=10.0)
    stats = compute_statistics("/bad-rate", results, 2000.0)
    thresholds = ThresholdsConfig(min_success_rate=0.99)
    try:
        assert_thresholds(stats, thresholds)
    except AssertionError as exc:
        assert "成功率" in str(exc)
        assert "bad-rate" in str(exc)
    else:
        raise AssertionError("Expected AssertionError but none raised")


def test_assert_thresholds_fail_latency() -> None:
    """验证延迟不达标时断言失败。"""
    results = _make_mock_results(10, fail_rate=0.0, base_latency=5000.0)
    stats = compute_statistics("/slow", results, 2000.0)
    thresholds = ThresholdsConfig(
        min_success_rate=0.0,
        max_p50_latency_ms=10.0,
        max_p95_latency_ms=10.0,
        max_p99_latency_ms=10.0,
    )
    try:
        assert_thresholds(stats, thresholds)
    except AssertionError as exc:
        msg = str(exc)
        assert "P50" in msg or "P95" in msg or "P99" in msg
    else:
        raise AssertionError("Expected AssertionError but none raised")
