# Vehicle Tools 单元测试 - 使用 SceneVehicleProvider,不依赖外部仿真器
# 运行指南: pytest tests/unit/test_vehicle_tools.py -v

from app.config import PolicyConfig
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.tools.vehicle import VehicleService, execute_vehicle_tool


def _make_service() -> VehicleService:
    """构造一个基于场景池提供者的 VehicleService。"""
    policy = PolicyConfig(temperature_min=18, temperature_max=30)
    return VehicleService(SceneVehicleProvider(get_scene_pool()), policy)


def test_get_vehicle_status_returns_state() -> None:
    """验证 get_vehicle_status 返回车速、电量、温度信息。"""
    service = _make_service()
    result = service.get_vehicle_status()
    assert result.success is True
    assert "车速" in result.output
    assert "电量" in result.output


def test_get_cabin_temperature_returns_value() -> None:
    """验证 get_cabin_temperature 返回温度数值。"""
    service = _make_service()
    result = service.get_cabin_temperature()
    assert result.success is True
    assert "车内温度" in result.output


def test_set_temperature_within_range_succeeds() -> None:
    """验证设置合法温度(24)成功。"""
    service = _make_service()
    result = service.set_temperature(24)
    assert result.success is True
    assert result.data["target_temperature_c"] == 24


def test_set_temperature_below_range_rejected() -> None:
    """验证低于下限(10)被 policy 拒绝。"""
    service = _make_service()
    result = service.set_temperature(10)
    assert result.success is False
    assert result.error == "policy_violation"


def test_set_temperature_above_range_rejected() -> None:
    """验证高于上限(40)被 policy 拒绝。"""
    service = _make_service()
    result = service.set_temperature(40)
    assert result.success is False
    assert result.error == "policy_violation"


def test_set_ac_toggles_state() -> None:
    """验证 set_ac 能切换空调状态。"""
    service = _make_service()
    result = service.set_ac(True)
    assert result.success is True
    assert result.data["ac_enabled"] is True


def test_execute_vehicle_tool_dispatch() -> None:
    """验证 execute_vehicle_tool 能正确分派到对应方法。"""
    service = _make_service()
    result = execute_vehicle_tool("get_vehicle_status", {}, service)
    assert result.success is True
    unknown = execute_vehicle_tool("not_a_tool", {}, service)
    assert unknown.success is False
