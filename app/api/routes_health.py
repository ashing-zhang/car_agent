# 健康检查路由
# 运行指南: 启动 API (python -m app.main) 后访问 http://localhost:8000/health

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """返回服务健康状态。"""
    return {"status": "ok", "service": "autoagent"}


@router.get("/health/ready")
async def readiness_check() -> dict:
    """返回服务就绪状态(基础实现,后续接入依赖检查)。"""
    return {"status": "ready", "service": "autoagent"}
