# FastAPI 应用入口
# 运行指南:
#   开发模式: python -m app.main
#   uvicorn 启动: uvicorn app.main:app --reload --port 8000
#   访问: http://localhost:8000/docs (Swagger UI)
#   指标: http://localhost:8000/metrics (Prometheus 抓取端点)
#   Dashboard: http://localhost:8000/api/v1/observability/dashboard

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.api.routes_agent import router as agent_router
from app.api.routes_health import router as health_router
from app.api.routes_observability import router as observability_router
from app.api.routes_vehicle import router as vehicle_router
from app.config import get_app_config, get_settings
from app.observability.logging import setup_logging
from app.observability.prometheus import render_prometheus
from app.observability.recorder import get_metrics_registry

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用实例。"""
    settings = get_settings()
    app_config = get_app_config()

    setup_logging(
        level=settings.log_level,
        structured=app_config.observability.structured_logging,
    )

    app = FastAPI(
        title="AutoAgent",
        description="Multimodal personalized automotive agent driven by a simulation scene pool",
        version="0.1.0",
    )

    app.state.settings = settings
    app.state.app_config = app_config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(vehicle_router, prefix="/api/v1")
    app.include_router(agent_router, prefix="/api/v1")
    app.include_router(observability_router, prefix="/api/v1")

    @app.get("/metrics", response_class=PlainTextResponse, tags=["observability"])
    async def root_metrics() -> str:
        """Prometheus 抓取端点(根路径,便于 scrape_configs 直接使用)。"""
        return render_prometheus(get_metrics_registry())

    logger.info(
        "FastAPI application created, environment=%s, structured_logging=%s",
        settings.environment, app_config.observability.structured_logging,
    )
    return app


app = create_app()


def main() -> None:
    """以编程方式启动 uvicorn 服务。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
