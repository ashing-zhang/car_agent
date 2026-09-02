# Vehicle API 路由
# 运行指南: 启动 API 后
#   GET  /api/v1/vehicle/status   查询车辆状态
#   POST /api/v1/vehicle/temperature  设置温度

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.tools.vehicle import get_vehicle_service

router = APIRouter(prefix="/vehicle", tags=["vehicle"])


class SetTemperatureRequest(BaseModel):
    """设置温度请求体。"""

    temperature_c: float = Field(ge=0, le=50)


class ToolResponse(BaseModel):
    """统一工具响应。"""

    success: bool
    tool_name: str
    output: str
    error: str | None = None


@router.get("/status", response_model=ToolResponse)
async def get_vehicle_status() -> ToolResponse:
    """返回当前车辆状态。"""
    result = get_vehicle_service().get_vehicle_status()
    return ToolResponse(success=result.success, tool_name=result.tool_name, output=result.output, error=result.error)


@router.post("/temperature", response_model=ToolResponse)
async def set_temperature(req: SetTemperatureRequest) -> ToolResponse:
    """设置目标车内温度(经 policy 校验)。"""
    result = get_vehicle_service().set_temperature(req.temperature_c)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.output)
    return ToolResponse(success=result.success, tool_name=result.tool_name, output=result.output, error=result.error)
