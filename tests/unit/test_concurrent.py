# 后端服务高并发测试 - 模拟多个用户同时访问 Agent 的场景
# 运行指南:
#   运行所有并发测试: pytest tests/unit/test_concurrent.py -v
#   指定标记运行: pytest tests/unit/test_concurrent.py -v -m "concurrent"
#   调整并发参数: 修改 configs/test_concurrent.yaml 配置文件
#   注意: 测试使用 MockLLMProvider,不依赖外部 LLM API Key

import asyncio
import logging
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from tests.unit.concurrent_utils import (
    ConcurrencyConfig,
    ConcurrentTestConfig,
    ConcurrencyStats,
    EndpointConfig,
    RequestResult,
    ThresholdsConfig,
    assert_thresholds,
    compute_statistics,
    generate_user_ids,
    load_concurrent_test_config,
    now_ms,
    pick_random_message,
    pick_random_temperature,
    random_jitter,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.concurrent


@pytest.fixture(scope="module")
def concurrent_config() -> ConcurrentTestConfig:
    """加载高并发测试配置(模块级共享)。"""
    return load_concurrent_test_config()


@pytest.fixture(scope="module")
def thresholds(concurrent_config: ConcurrentTestConfig) -> ThresholdsConfig:
    """获取测试通过阈值。"""
    return concurrent_config.thresholds


@pytest.fixture(scope="module")
def concurrency(concurrent_config: ConcurrentTestConfig) -> ConcurrencyConfig:
    """获取并发参数配置。"""
    return concurrent_config.concurrency


async def _run_concurrent_get(
    endpoint_cfg: EndpointConfig,
    concurrency_cfg: ConcurrencyConfig,
    thresholds_cfg: ThresholdsConfig,
) -> ConcurrencyStats:
    """通用 GET 端点并发测试执行器。"""
    return await _run_concurrent_requests(
        endpoint_cfg=endpoint_cfg,
        concurrency_cfg=concurrency_cfg,
        thresholds_cfg=thresholds_cfg,
        build_request=_build_get_request,
    )


async def _run_concurrent_requests(
    endpoint_cfg: EndpointConfig,
    concurrency_cfg: ConcurrencyConfig,
    thresholds_cfg: ThresholdsConfig,
    build_request: Any,
) -> ConcurrencyStats:
    """通用并发请求执行器:创建 N 用户 x M 请求的并发任务,收集并统计结果。"""
    if not endpoint_cfg.enabled:
        logger.info("端点 %s 已在配置中禁用,跳过测试", endpoint_cfg.path)
        return ConcurrencyStats(endpoint=endpoint_cfg.path)

    user_ids = generate_user_ids(concurrency_cfg.concurrent_users)
    start_time = now_ms()
    results: list[RequestResult] = []

    async def _user_session(user_id: str, session_idx: int) -> list[RequestResult]:
        """单个用户的会话:连续发起 requests_per_user 次请求(带随机抖动)。"""
        session_id = f"session-{user_id}-{session_idx:02d}"
        user_results: list[RequestResult] = []
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
            timeout=concurrency_cfg.request_timeout,
        ) as client:
            for req_idx in range(concurrency_cfg.requests_per_user):
                jitter = random_jitter(
                    concurrency_cfg.request_jitter_min,
                    concurrency_cfg.request_jitter_max,
                )
                if jitter > 0:
                    await asyncio.sleep(jitter)
                try:
                    req_start = now_ms()
                    response = await build_request(client, endpoint_cfg, user_id, req_idx)
                    latency = now_ms() - req_start
                    success = 200 <= response.status_code < 500
                    err = None if success else f"HTTP {response.status_code}"
                    user_results.append(RequestResult(
                        user_id=user_id,
                        session_id=session_id,
                        request_index=req_idx,
                        success=success,
                        status_code=response.status_code,
                        latency_ms=latency,
                        error_message=err,
                        endpoint=endpoint_cfg.path,
                    ))
                except Exception as exc:  # noqa: BLE001
                    latency = now_ms() - req_start
                    err_msg = f"{type(exc).__name__}: {exc}"
                    logger.warning("请求失败 user=%s req=%d: %s", user_id, req_idx, err_msg)
                    user_results.append(RequestResult(
                        user_id=user_id,
                        session_id=session_id,
                        request_index=req_idx,
                        success=False,
                        status_code=0,
                        latency_ms=latency,
                        error_message=err_msg,
                        endpoint=endpoint_cfg.path,
                    ))
        return user_results

    tasks = [
        _user_session(uid, idx)
        for idx, uid in enumerate(user_ids)
    ]

    session_results = await asyncio.gather(*tasks, return_exceptions=False)
    for sr in session_results:
        results.extend(sr)

    total_duration = now_ms() - start_time
    stats = compute_statistics(endpoint_cfg.path, results, total_duration)
    logger.info("并发测试完成: %s", stats.summary())
    assert_thresholds(stats, thresholds_cfg)
    return stats


async def _build_get_request(client: AsyncClient, endpoint_cfg: EndpointConfig,
                             user_id: str, req_idx: int) -> Any:
    """构造并发送 GET 请求。"""
    return await client.get(endpoint_cfg.path)


async def _build_vehicle_temp_request(client: AsyncClient, endpoint_cfg: EndpointConfig,
                                      user_id: str, req_idx: int) -> Any:
    """构造并发送车辆设置温度 POST 请求。"""
    temp = pick_random_temperature(endpoint_cfg.temperature_values)
    return await client.post(
        endpoint_cfg.path,
        json={"temperature_c": temp},
    )


async def _build_agent_chat_request(client: AsyncClient, endpoint_cfg: EndpointConfig,
                                    user_id: str, req_idx: int) -> Any:
    """构造并发送 Agent Chat POST 请求。"""
    message = pick_random_message(endpoint_cfg.test_messages)
    payload = {
        "user_id": user_id,
        "session_id": f"chat-{user_id}-{req_idx:03d}",
        "message": message,
    }
    return await client.post(endpoint_cfg.path, json=payload)


async def _build_agent_plan_request(client: AsyncClient, endpoint_cfg: EndpointConfig,
                                    user_id: str, req_idx: int) -> Any:
    """构造并发送 Agent Plan POST 请求。"""
    message = pick_random_message(endpoint_cfg.test_messages)
    payload = {
        "user_id": user_id,
        "session_id": f"plan-{user_id}-{req_idx:03d}",
        "message": message,
    }
    return await client.post(endpoint_cfg.path, json=payload)


# ============== 具体测试用例 ==============

@pytest.mark.asyncio
async def test_concurrent_health_endpoint(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发测试:多个用户同时访问健康检查端点 /api/v1/health。"""
    stats = await _run_concurrent_get(
        concurrent_config.health_endpoint, concurrency, thresholds,
    )
    if concurrent_config.health_endpoint.enabled:
        assert stats.total_requests == concurrency.concurrent_users * concurrency.requests_per_user


@pytest.mark.asyncio
async def test_concurrent_readiness_endpoint(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发测试:多个用户同时访问就绪检查端点 /api/v1/health/ready。"""
    stats = await _run_concurrent_get(
        concurrent_config.readiness_endpoint, concurrency, thresholds,
    )
    if concurrent_config.readiness_endpoint.enabled:
        assert stats.total_requests == concurrency.concurrent_users * concurrency.requests_per_user


@pytest.mark.asyncio
async def test_concurrent_vehicle_status_endpoint(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发测试:多个用户同时查询车辆状态 /api/v1/vehicle/status。"""
    stats = await _run_concurrent_get(
        concurrent_config.vehicle_status_endpoint, concurrency, thresholds,
    )
    if concurrent_config.vehicle_status_endpoint.enabled:
        assert stats.total_requests == concurrency.concurrent_users * concurrency.requests_per_user


@pytest.mark.asyncio
async def test_concurrent_vehicle_temperature_endpoint(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发测试:多个用户同时设置车内温度 /api/v1/vehicle/temperature。"""
    cfg = concurrent_config.vehicle_temperature_endpoint
    if not cfg.enabled:
        pytest.skip("vehicle_temperature endpoint disabled in config")
    stats = await _run_concurrent_requests(
        endpoint_cfg=cfg,
        concurrency_cfg=concurrency,
        thresholds_cfg=thresholds,
        build_request=_build_vehicle_temp_request,
    )
    assert stats.total_requests == concurrency.concurrent_users * concurrency.requests_per_user


@pytest.mark.asyncio
async def test_concurrent_agent_chat_endpoint(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发测试:多个用户同时与 Agent 聊天 /api/v1/agent/chat (MockLLM)。"""
    cfg = concurrent_config.agent_chat_endpoint
    if not cfg.enabled:
        pytest.skip("agent_chat endpoint disabled in config")
    stats = await _run_concurrent_requests(
        endpoint_cfg=cfg,
        concurrency_cfg=concurrency,
        thresholds_cfg=thresholds,
        build_request=_build_agent_chat_request,
    )
    assert stats.total_requests == concurrency.concurrent_users * concurrency.requests_per_user


@pytest.mark.asyncio
async def test_concurrent_agent_plan_endpoint(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发测试:多个用户同时请求 Agent 规划任务 /api/v1/agent/plan (MockLLM)。"""
    cfg = concurrent_config.agent_plan_endpoint
    if not cfg.enabled:
        pytest.skip("agent_plan endpoint disabled in config")
    stats = await _run_concurrent_requests(
        endpoint_cfg=cfg,
        concurrency_cfg=concurrency,
        thresholds_cfg=thresholds,
        build_request=_build_agent_plan_request,
    )
    assert stats.total_requests == concurrency.concurrent_users * concurrency.requests_per_user


@pytest.mark.asyncio
async def test_concurrent_multi_endpoint_mixed(
    concurrent_config: ConcurrentTestConfig,
    concurrency: ConcurrencyConfig,
    thresholds: ThresholdsConfig,
) -> None:
    """高并发混合测试:不同用户同时访问不同类型端点,模拟真实混合流量场景。"""
    enabled_endpoints = [
        cfg for cfg in [
            concurrent_config.health_endpoint,
            concurrent_config.vehicle_status_endpoint,
            concurrent_config.agent_chat_endpoint,
        ] if cfg.enabled
    ]
    if len(enabled_endpoints) < 2:
        pytest.skip("need at least 2 enabled endpoints for mixed test")

    mixed_stats: list[ConcurrencyStats] = []
    user_per_endpoint = max(1, concurrency.concurrent_users // len(enabled_endpoints))
    concurrency_split = ConcurrencyConfig(
        concurrent_users=user_per_endpoint,
        requests_per_user=concurrency.requests_per_user,
        request_jitter_min=concurrency.request_jitter_min,
        request_jitter_max=concurrency.request_jitter_max,
        overall_timeout=concurrency.overall_timeout,
        request_timeout=concurrency.request_timeout,
    )

    async def _run_one(ep_cfg: EndpointConfig) -> ConcurrencyStats:
        if ep_cfg.method.upper() == "GET":
            return await _run_concurrent_get(ep_cfg, concurrency_split, thresholds)
        if "chat" in ep_cfg.path:
            return await _run_concurrent_requests(
                ep_cfg, concurrency_split, thresholds, _build_agent_chat_request,
            )
        return await _run_concurrent_requests(
            ep_cfg, concurrency_split, thresholds, _build_get_request,
        )

    mixed_stats = list(await asyncio.gather(*(_run_one(ep) for ep in enabled_endpoints)))

    for stats in mixed_stats:
        if stats.total_requests > 0:
            logger.info("混合流量结果: %s", stats.summary())
