# 高并发测试辅助工具模块
# 运行指南:
#   作为测试辅助模块被导入: from tests.unit.concurrent_utils import ...
#   提供类型定义、配置加载、统计计算、并发任务生成等核心能力

import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


class ConcurrencyConfig(BaseModel):
    """并发测试基础参数配置。"""

    concurrent_users: int = Field(default=20, ge=1, le=500)
    requests_per_user: int = Field(default=5, ge=1, le=100)
    request_jitter_min: float = Field(default=0.0, ge=0.0)
    request_jitter_max: float = Field(default=0.1, ge=0.0)
    overall_timeout: int = Field(default=120, ge=1)
    request_timeout: int = Field(default=30, ge=1)


class ThresholdsConfig(BaseModel):
    """测试通过阈值配置。"""

    min_success_rate: float = Field(default=0.95, ge=0.0, le=1.0)
    max_p50_latency_ms: float = Field(default=2000.0, ge=0.0)
    max_p95_latency_ms: float = Field(default=5000.0, ge=0.0)
    max_p99_latency_ms: float = Field(default=10000.0, ge=0.0)


class EndpointConfig(BaseModel):
    """单端点测试配置。"""

    enabled: bool = True
    path: str
    method: str = "GET"
    test_messages: list[str] | None = None
    temperature_values: list[float] | None = None


class ConcurrentTestConfig(BaseModel):
    """高并发测试完整配置(配置驱动)。"""

    concurrency: ConcurrencyConfig = ConcurrencyConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    health_endpoint: EndpointConfig
    readiness_endpoint: EndpointConfig
    vehicle_status_endpoint: EndpointConfig
    vehicle_temperature_endpoint: EndpointConfig
    agent_chat_endpoint: EndpointConfig
    agent_plan_endpoint: EndpointConfig


@dataclass
class RequestResult:
    """单次请求结果记录。"""

    user_id: str
    session_id: str
    request_index: int
    success: bool
    status_code: int
    latency_ms: float
    error_message: str | None = None
    endpoint: str = ""


@dataclass
class ConcurrencyStats:
    """并发测试统计结果。"""

    endpoint: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    total_duration_ms: float = 0.0
    requests_per_second: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """生成可读的统计摘要文本。"""
        return (
            f"[{self.endpoint}] 总请求={self.total_requests}, "
            f"成功={self.successful_requests}, 失败={self.failed_requests}, "
            f"成功率={self.success_rate:.2%}, "
            f"延迟(ms): min={self.latency_min_ms:.1f}, mean={self.latency_mean_ms:.1f}, "
            f"P50={self.latency_p50_ms:.1f}, P95={self.latency_p95_ms:.1f}, P99={self.latency_p99_ms:.1f}, "
            f"QPS={self.requests_per_second:.2f}"
        )


def load_concurrent_test_config(config_file: str = "test_concurrent.yaml") -> ConcurrentTestConfig:
    """加载高并发测试 yaml 配置文件。"""
    config_path = CONFIGS_DIR / config_file
    if not config_path.exists():
        logger.warning("并发测试配置文件不存在 %s,使用默认配置", config_path)
        return ConcurrentTestConfig(
            health_endpoint=EndpointConfig(path="/api/v1/health"),
            readiness_endpoint=EndpointConfig(path="/api/v1/health/ready"),
            vehicle_status_endpoint=EndpointConfig(path="/api/v1/vehicle/status"),
            vehicle_temperature_endpoint=EndpointConfig(
                path="/api/v1/vehicle/temperature",
                method="POST",
                temperature_values=[22.0, 24.0, 26.0],
            ),
            agent_chat_endpoint=EndpointConfig(
                path="/api/v1/agent/chat",
                method="POST",
                test_messages=["你好", "车速多少"],
            ),
            agent_plan_endpoint=EndpointConfig(
                path="/api/v1/agent/plan",
                method="POST",
                test_messages=["规划路线"],
            ),
        )
    with config_path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}
    config = ConcurrentTestConfig(**raw_data)
    logger.info("加载并发测试配置: %s, 并发用户=%d, 每用户请求=%d",
                config_path, config.concurrency.concurrent_users,
                config.concurrency.requests_per_user)
    return config


def compute_statistics(endpoint: str, results: list[RequestResult],
                       total_duration_ms: float) -> ConcurrencyStats:
    """根据请求结果列表计算统计指标。"""
    stats = ConcurrencyStats(endpoint=endpoint)
    stats.total_requests = len(results)
    if stats.total_requests == 0:
        return stats

    stats.successful_requests = sum(1 for r in results if r.success)
    stats.failed_requests = stats.total_requests - stats.successful_requests
    stats.success_rate = stats.successful_requests / stats.total_requests

    latencies = [r.latency_ms for r in results if r.success]
    if latencies:
        stats.latency_min_ms = min(latencies)
        stats.latency_max_ms = max(latencies)
        stats.latency_mean_ms = statistics.mean(latencies)
        sorted_lat = sorted(latencies)
        stats.latency_p50_ms = _percentile(sorted_lat, 50)
        stats.latency_p95_ms = _percentile(sorted_lat, 95)
        stats.latency_p99_ms = _percentile(sorted_lat, 99)

    stats.errors = [r.error_message for r in results if not r.success and r.error_message]
    stats.total_duration_ms = total_duration_ms
    if total_duration_ms > 0:
        stats.requests_per_second = (stats.total_requests * 1000.0) / total_duration_ms

    return stats


def _percentile(sorted_data: list[float], pct: float) -> float:
    """计算已排序数据的百分位数(线性插值法)。"""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_data[f]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def assert_thresholds(stats: ConcurrencyStats, thresholds: ThresholdsConfig) -> None:
    """根据配置阈值断言测试是否通过,失败时 AssertionError 包含详细信息。"""
    errors: list[str] = []
    if stats.success_rate < thresholds.min_success_rate:
        errors.append(
            f"成功率 {stats.success_rate:.2%} 低于阈值 {thresholds.min_success_rate:.2%}"
        )
    if stats.latency_p50_ms > thresholds.max_p50_latency_ms:
        errors.append(
            f"P50 延迟 {stats.latency_p50_ms:.1f}ms 超过阈值 {thresholds.max_p50_latency_ms:.1f}ms"
        )
    if stats.latency_p95_ms > thresholds.max_p95_latency_ms:
        errors.append(
            f"P95 延迟 {stats.latency_p95_ms:.1f}ms 超过阈值 {thresholds.max_p95_latency_ms:.1f}ms"
        )
    if stats.latency_p99_ms > thresholds.max_p99_latency_ms:
        errors.append(
            f"P99 延迟 {stats.latency_p99_ms:.1f}ms 超过阈值 {thresholds.max_p99_latency_ms:.1f}ms"
        )
    if errors:
        detail = "; ".join(errors)
        raise AssertionError(f"[{stats.endpoint}] 并发测试阈值不满足: {detail}\n{stats.summary()}")


def generate_user_ids(count: int) -> list[str]:
    """生成指定数量的唯一用户 ID。"""
    return [f"concurrent-user-{i:04d}" for i in range(count)]


def random_jitter(min_sec: float, max_sec: float) -> float:
    """生成随机延迟秒数(用于模拟用户请求间的自然间隔)。"""
    if max_sec <= min_sec:
        return min_sec
    return random.uniform(min_sec, max_sec)


def pick_random_message(messages: list[str] | None) -> str:
    """从消息池中随机选取一条消息。"""
    if not messages:
        return "你好"
    return random.choice(messages)


def pick_random_temperature(values: list[float] | None) -> float:
    """从温度池随机选取一个温度值。"""
    if not values:
        return 24.0
    return random.choice(values)


def now_ms() -> float:
    """返回当前高精度时间戳(毫秒)。"""
    return time.perf_counter() * 1000.0
