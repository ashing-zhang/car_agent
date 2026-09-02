# Phase 0 基础健康检查单元测试
# 运行指南: pytest tests/unit/test_health.py -v

from fastapi.testclient import TestClient

from app.main import app


def test_health_check() -> None:
    """验证健康检查端点返回 ok。"""
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_check() -> None:
    """验证就绪检查端点返回 ready。"""
    client = TestClient(app)
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
