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


# ---------------- 座椅控制测试 ----------------


def test_set_seat_position_valid_driver() -> None:
    """验证设置驾驶位座椅位置成功。"""
    service = _make_service()
    result = service.set_seat_position("driver", slide_percent=60, backrest_angle_deg=100)
    assert result.success is True
    assert result.data["seat"] == "driver"
    assert result.data["slide_percent"] == 60
    assert result.data["backrest_angle_deg"] == 100


def test_set_seat_position_invalid_seat_rejected() -> None:
    """验证无效座椅标识被拒绝。"""
    service = _make_service()
    result = service.set_seat_position("rear", slide_percent=50)
    assert result.success is False
    assert result.error == "invalid_seat"


def test_set_seat_position_slide_out_of_range_rejected() -> None:
    """验证滑动百分比超出范围被 policy 拒绝。"""
    service = _make_service()
    result = service.set_seat_position("driver", slide_percent=150)
    assert result.success is False
    assert result.error == "policy_violation"


def test_set_seat_position_backrest_out_of_range_rejected() -> None:
    """验证靠背角度超出范围被 policy 拒绝。"""
    service = _make_service()
    result = service.set_seat_position("passenger", backrest_angle_deg=270)
    assert result.success is False
    assert result.error == "policy_violation"


def test_set_seat_position_missing_args_rejected() -> None:
    """验证未指定任何参数时被拒绝。"""
    service = _make_service()
    result = service.set_seat_position("driver")
    assert result.success is False
    assert result.error == "missing_args"


def test_set_seat_ventilation_valid_level() -> None:
    """验证设置座椅通风档位成功。"""
    service = _make_service()
    result = service.set_seat_ventilation("driver", 2)
    assert result.success is True
    assert result.data["level"] == 2
    assert "2 档" in result.output


def test_set_seat_ventilation_level_zero_disables() -> None:
    """验证档位 0 关闭通风。"""
    service = _make_service()
    result = service.set_seat_ventilation("passenger", 0)
    assert result.success is True
    assert "关闭" in result.output


def test_set_seat_ventilation_out_of_range_rejected() -> None:
    """验证通风档位超出范围被拒绝。"""
    service = _make_service()
    result = service.set_seat_ventilation("driver", 5)
    assert result.success is False
    assert result.error == "policy_violation"


def test_set_seat_massage_valid_level() -> None:
    """验证设置座椅按摩档位成功。"""
    service = _make_service()
    result = service.set_seat_massage("passenger", 3)
    assert result.success is True
    assert result.data["level"] == 3


def test_set_seat_massage_invalid_seat_rejected() -> None:
    """验证按摩无效座椅标识被拒绝。"""
    service = _make_service()
    result = service.set_seat_massage("driver2", 1)
    assert result.success is False
    assert result.error == "invalid_seat"


def test_set_seat_heating_valid_level() -> None:
    """验证设置座椅加热档位成功。"""
    service = _make_service()
    result = service.set_seat_heating("driver", 1)
    assert result.success is True
    assert result.data["seat"] == "driver"


def test_set_seat_heating_out_of_range_rejected() -> None:
    """验证加热档位超出范围被拒绝。"""
    service = _make_service()
    result = service.set_seat_heating("passenger", -1)
    assert result.success is False
    assert result.error == "policy_violation"


# ---------------- 车窗 / 后备箱测试 ----------------


def test_set_window_all_windows() -> None:
    """验证设置所有车窗开启成功。"""
    service = _make_service()
    result = service.set_window("all", 50)
    assert result.success is True
    assert result.data["window"] == "all"
    assert result.data["open_percent"] == 50


def test_set_window_single_window() -> None:
    """验证设置单个车窗开启成功。"""
    service = _make_service()
    result = service.set_window("driver_front", 100)
    assert result.success is True
    assert result.data["window"] == "driver_front"


def test_set_window_close_all() -> None:
    """验证关闭所有车窗。"""
    service = _make_service()
    result = service.set_window("all", 0)
    assert result.success is True
    assert "完全关闭" in result.output


def test_set_window_invalid_window_rejected() -> None:
    """验证无效车窗标识被拒绝。"""
    service = _make_service()
    result = service.set_window("sunroof", 50)
    assert result.success is False
    assert result.error == "invalid_window"


def test_set_window_out_of_range_rejected() -> None:
    """验证车窗开启百分比超出范围被拒绝。"""
    service = _make_service()
    result = service.set_window("driver_front", 150)
    assert result.success is False
    assert result.error == "policy_violation"


def test_set_trunk_open() -> None:
    """验证开启后备箱成功。"""
    service = _make_service()
    result = service.set_trunk(True)
    assert result.success is True
    assert result.data["trunk_open"] is True
    assert "开启" in result.output


def test_set_trunk_close() -> None:
    """验证关闭后备箱成功。"""
    service = _make_service()
    result = service.set_trunk(False)
    assert result.success is True
    assert result.data["trunk_open"] is False
    assert "关闭" in result.output


# ---------------- 雨刮 / 车灯测试 ----------------


def test_set_wiper_all_levels() -> None:
    """验证设置所有雨刮档位成功。"""
    service = _make_service()
    for level in ["off", "low", "medium", "high", "auto"]:
        result = service.set_wiper(level)
        assert result.success is True
        assert result.data["wiper_level"] == level


def test_set_wiper_invalid_level_rejected() -> None:
    """验证无效雨刮档位被拒绝。"""
    service = _make_service()
    result = service.set_wiper("ultra")
    assert result.success is False
    assert result.error == "invalid_wiper_level"


def test_set_light_all_modes() -> None:
    """验证设置所有车灯模式成功。"""
    service = _make_service()
    for mode in ["off", "parking", "low_beam", "high_beam", "auto"]:
        result = service.set_light(mode)
        assert result.success is True
        assert result.data["light_mode"] == mode


def test_set_light_invalid_mode_rejected() -> None:
    """验证无效车灯模式被拒绝。"""
    service = _make_service()
    result = service.set_light("fog")
    assert result.success is False
    assert result.error == "invalid_light_mode"


# ---------------- execute_vehicle_tool 扩展分派测试 ----------------


def test_execute_vehicle_tool_dispatch_new_tools() -> None:
    """验证 execute_vehicle_tool 能正确分派新工具。"""
    service = _make_service()
    r1 = execute_vehicle_tool("set_seat_ventilation", {"seat": "driver", "level": 2}, service)
    assert r1.success is True
    r2 = execute_vehicle_tool("set_window", {"window": "all", "open_percent": 30}, service)
    assert r2.success is True
    r3 = execute_vehicle_tool("set_wiper", {"level": "auto"}, service)
    assert r3.success is True
    r4 = execute_vehicle_tool("set_light", {"mode": "low_beam"}, service)
    assert r4.success is True
    r5 = execute_vehicle_tool("set_trunk", {"open": True}, service)
    assert r5.success is True
    r6 = execute_vehicle_tool("set_seat_position", {"seat": "driver", "slide_percent": 70}, service)
    assert r6.success is True
    r7 = execute_vehicle_tool("set_seat_massage", {"seat": "passenger", "level": 1}, service)
    assert r7.success is True
    r8 = execute_vehicle_tool("set_seat_heating", {"seat": "driver", "level": 0}, service)
    assert r8.success is True
