# Scene Pool 单元测试 - 场景池随机选景与 Provider 覆盖逻辑
# 运行指南: pytest tests/unit/test_scene_pool.py -v

from app.context.schemas import EnvironmentState, VehicleState
from app.simulation.providers import SceneEnvironmentProvider, SceneVehicleProvider
from app.simulation.scene_pool import SceneDefinition, ScenePool, get_scene_pool
from app.tools.environment import EnvironmentService


def _make_pool() -> ScenePool:
    """构造固定场景池(确定性随机源)。"""
    scenes = [
        SceneDefinition(
            name="a",
            vehicle={"speed_kmh": 30.0, "cabin_temperature_c": 21.0, "ac_enabled": False},
            environment={"weather": "rain", "traffic_density": "high"},
        ),
        SceneDefinition(
            name="b",
            vehicle={"speed_kmh": 90.0, "cabin_temperature_c": 25.0, "ac_enabled": True},
            environment={"weather": "clear", "traffic_density": "low"},
        ),
    ]
    return ScenePool(scenes)


def test_scene_pool_pick_returns_scene() -> None:
    """验证 pick 返回池内场景。"""
    pool = _make_pool()
    scene = pool.pick()
    assert scene.name in {"a", "b"}
    assert pool.size == 2


def test_scene_pool_empty_rejected() -> None:
    """验证空场景池被拒绝。"""
    try:
        ScenePool([])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_scene_pool_from_config() -> None:
    """验证从 configs/scene_pool.yaml 加载的场景池非空。"""
    pool = get_scene_pool()
    assert pool.size >= 1


def test_scene_vehicle_provider_returns_vehicle_state() -> None:
    """验证 SceneVehicleProvider 返回合法 VehicleState。"""
    provider = SceneVehicleProvider(_make_pool())
    state = provider.get_state()
    assert isinstance(state, VehicleState)
    assert state.speed_kmh >= 0
    assert 0 <= state.battery_percent <= 100


def test_scene_vehicle_provider_control_overrides() -> None:
    """验证温度/空调控制覆盖随机场景值。"""
    provider = SceneVehicleProvider(_make_pool())
    provider.set_cabin_temperature(24)
    provider.set_ac(False)
    state = provider.get_state()
    assert state.target_temperature_c == 24
    assert state.cabin_temperature_c == 24
    assert state.ac_enabled is False


def test_scene_environment_provider_returns_environment_state() -> None:
    """验证 SceneEnvironmentProvider 返回合法 EnvironmentState。"""
    provider = SceneEnvironmentProvider(_make_pool())
    state = provider.get_state()
    assert isinstance(state, EnvironmentState)
    assert state.weather in {"rain", "clear"}


def test_environment_service_with_scene_provider() -> None:
    """验证 EnvironmentService 委托场景池并正确缓存/刷新。"""
    service = EnvironmentService(SceneEnvironmentProvider(_make_pool()))
    result = service.get_weather()
    assert result.success
    assert "天气" in result.output
    scene = service.get_camera_scene()
    assert scene.success
    assert scene.data["weather"] in {"rain", "clear"}
