# Vehicle Tools - 车辆状态查询与控制工具(规格第9节)
# 运行指南:
#   分层: Agent Tool → VehicleService → VehicleProvider(委托) → VehicleStateStore + 场景池(app/simulation)
#   偏好记录: set_temperature/set_ac 成功后,若会话上下文已设置且无同类偏好,自动写入低置信度偏好(MemoryService)
#   所有工具经 Schema Validation → Policy Validation → Execution → Audit Log(规格第10节)
#   工厂: from app.tools.vehicle import get_vehicle_service

import logging

from app.config import PolicyConfig, get_app_config
from app.context.providers import VehicleProvider
from app.simulation.providers import SceneVehicleProvider
from app.simulation.scene_pool import get_scene_pool
from app.simulation.vehicle_state_store import get_session_context
from app.tools.schemas import ToolResult

logger = logging.getLogger(__name__)


class VehicleService:
    """车辆服务层,负责 policy 校验、偏好同步后委托给 VehicleProvider 执行。"""

    def __init__(
        self,
        provider: VehicleProvider,
        policy: PolicyConfig,
        *,
        memory_service: object | None = None,
        auto_save_preference_confidence: float | None = None,
    ) -> None:
        """注入车辆状态提供者、安全策略与可选的记忆服务。"""
        self._provider = provider
        self._policy = policy
        self._memory_service = memory_service
        if auto_save_preference_confidence is None:
            auto_save_preference_confidence = 0.55
        self._auto_save_confidence = auto_save_preference_confidence

    # ---------------- 偏好辅助 ----------------

    def _lazy_memory_service(self) -> object | None:
        """惰性获取 MemoryService,获取失败返回 None(调用方跳过偏好记录)。"""
        if self._memory_service is not None:
            return self._memory_service
        try:
            from app.memory.service import get_memory_service
            return get_memory_service()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MemoryService unavailable for preference sync: %s", exc)
            return None

    def _maybe_save_temperature_preference(self, temperature_c: float) -> None:
        """若已设置会话上下文且无相同/更高置信度偏好,则写入低置信度偏好。"""
        user_id, session_id = get_session_context()
        if not user_id or not session_id:
            return
        service = self._lazy_memory_service()
        if service is None:
            return
        try:
            existing = service.recall_preference(user_id, "preferred_temperature")
            if existing is not None:
                try:
                    existing_val = float(existing.value)
                    if abs(existing_val - temperature_c) < 0.5:
                        return
                except (TypeError, ValueError):
                    pass
                if existing.confidence >= self._auto_save_confidence:
                    return
            from app.memory.models import Memory, MemoryType
            candidate = Memory(
                user_id=user_id,
                session_id=session_id,
                type=MemoryType.PREFERENCE,
                predicate="preferred_temperature",
                object=f"{temperature_c}celsius",
                value=temperature_c,
                unit="celsius",
                confidence=self._auto_save_confidence,
                source="tool_action",
                metadata={"reason": "auto_saved_by_set_temperature"},
            )
            service.remember(f"我习惯车内温度{temperature_c}度", user_id, session_id)
            logger.info(
                "Auto-saved temperature preference: user=%s temp=%.1f°C confidence=%.2f",
                user_id, temperature_c, self._auto_save_confidence,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Auto-save temperature preference failed (user=%s)", user_id)

    def _maybe_save_ac_mode_preference(self, enabled: bool) -> None:
        """同理,用户每次显式开/关空调时,更新 ac_mode 偏好为 auto / fan(弱置信度)。"""
        user_id, session_id = get_session_context()
        if not user_id or not session_id:
            return
        service = self._lazy_memory_service()
        if service is None:
            return
        try:
            existing = service.recall_preference(user_id, "ac_mode")
            if existing is not None and existing.confidence >= self._auto_save_confidence:
                return
            if enabled:
                service.remember("我习惯使用自动空调模式", user_id, session_id)
            logger.info(
                "Auto-saved AC preference hint: user=%s enabled=%s",
                user_id, enabled,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Auto-save AC preference failed (user=%s)", user_id)

    # ---------------- Tool 方法 ----------------

    def get_vehicle_status(self) -> ToolResult:
        """查询当前车辆状态。"""
        try:
            state = self._provider.get_state()
            output = (
                f"当前车速 {state.speed_kmh:.0f} km/h,电量 {state.battery_percent:.0f}%,"
                f"车内温度 {state.cabin_temperature_c:.1f}℃。"
            )
            logger.info("get_vehicle_status: %s", output)
            return ToolResult(
                success=True,
                tool_name="get_vehicle_status",
                output=output,
                data=state.model_dump(mode="json"),
            )
        except Exception as exc:
            logger.exception("get_vehicle_status failed")
            return ToolResult(success=False, tool_name="get_vehicle_status", output="", error=str(exc))

    def get_cabin_temperature(self) -> ToolResult:
        """查询当前车内温度。"""
        try:
            state = self._provider.get_state()
            output = f"当前车内温度 {state.cabin_temperature_c:.1f}℃。"
            return ToolResult(
                success=True,
                tool_name="get_cabin_temperature",
                output=output,
                data={"cabin_temperature_c": state.cabin_temperature_c},
            )
        except Exception as exc:
            logger.exception("get_cabin_temperature failed")
            return ToolResult(success=False, tool_name="get_cabin_temperature", output="", error=str(exc))

    def set_temperature(self, temperature_c: float) -> ToolResult:
        """设置目标车内温度,执行前进行 policy 校验(18-30℃),成功后弱同步到偏好记忆。"""
        if not (self._policy.temperature_min <= temperature_c <= self._policy.temperature_max):
            msg = (
                f"温度 {temperature_c}°C 超出安全范围 "
                f"[{self._policy.temperature_min}, {self._policy.temperature_max}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_temperature", output=msg, error="policy_violation")
        try:
            self._provider.set_cabin_temperature(temperature_c)
            output = f"已将目标车内温度设置为 {temperature_c}°C。"
            logger.info("set_temperature(%.1f) -> %s", temperature_c, output)
            self._maybe_save_temperature_preference(temperature_c)
            return ToolResult(
                success=True,
                tool_name="set_temperature",
                output=output,
                data={"target_temperature_c": temperature_c},
            )
        except Exception as exc:
            logger.exception("set_temperature failed")
            return ToolResult(success=False, tool_name="set_temperature", output="", error=str(exc))

    def set_ac(self, enabled: bool) -> ToolResult:
        """开启或关闭空调,成功后弱同步到偏好记忆(仅开启时建议 auto 模式)。"""
        try:
            self._provider.set_ac(enabled)
            output = f"空调已{'开启' if enabled else '关闭'}。"
            logger.info("set_ac(%s) -> %s", enabled, output)
            self._maybe_save_ac_mode_preference(enabled)
            return ToolResult(
                success=True,
                tool_name="set_ac",
                output=output,
                data={"ac_enabled": enabled},
            )
        except Exception as exc:
            logger.exception("set_ac failed")
            return ToolResult(success=False, tool_name="set_ac", output="", error=str(exc))

    # ---------------- 座椅控制 ----------------

    def set_seat_position(
        self,
        seat: str,
        slide_percent: int | None = None,
        backrest_angle_deg: int | None = None,
    ) -> ToolResult:
        """设置座椅位置,执行 policy 校验(座椅参数范围)。"""
        seat = seat.lower()
        if seat not in {"driver", "passenger"}:
            msg = f"未知座椅标识 {seat}(支持: driver, passenger)。"
            logger.warning("Invalid seat: %s", seat)
            return ToolResult(success=False, tool_name="set_seat_position", output=msg, error="invalid_seat")
        if slide_percent is not None and not (
            self._policy.seat_slide_min <= slide_percent <= self._policy.seat_slide_max
        ):
            msg = (
                f"座椅滑动百分比 {slide_percent} 超出安全范围 "
                f"[{self._policy.seat_slide_min}, {self._policy.seat_slide_max}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_seat_position", output=msg, error="policy_violation")
        if backrest_angle_deg is not None and not (
            self._policy.seat_backrest_min_deg <= backrest_angle_deg <= self._policy.seat_backrest_max_deg
        ):
            msg = (
                f"靠背角度 {backrest_angle_deg}° 超出安全范围 "
                f"[{self._policy.seat_backrest_min_deg}, {self._policy.seat_backrest_max_deg}]°。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_seat_position", output=msg, error="policy_violation")
        if slide_percent is None and backrest_angle_deg is None:
            msg = "至少指定 slide_percent 或 backrest_angle_deg 之一。"
            logger.warning("set_seat_position missing arguments")
            return ToolResult(success=False, tool_name="set_seat_position", output=msg, error="missing_args")
        try:
            self._provider.set_seat_position(seat, slide_percent, backrest_angle_deg)
            seat_cn = "驾驶位" if seat == "driver" else "副驾驶位"
            parts: list[str] = []
            if slide_percent is not None:
                parts.append(f"滑动 {slide_percent}%")
            if backrest_angle_deg is not None:
                parts.append(f"靠背角度 {backrest_angle_deg}°")
            output = f"已将{seat_cn}调整为:" + ", ".join(parts) + "。"
            logger.info("set_seat_position(%s) -> %s", seat, output)
            return ToolResult(
                success=True,
                tool_name="set_seat_position",
                output=output,
                data={
                    "seat": seat,
                    "slide_percent": slide_percent,
                    "backrest_angle_deg": backrest_angle_deg,
                },
            )
        except Exception as exc:
            logger.exception("set_seat_position failed")
            return ToolResult(success=False, tool_name="set_seat_position", output="", error=str(exc))

    def set_seat_ventilation(self, seat: str, level: int) -> ToolResult:
        """设置座椅通风档位(0-3)。"""
        seat = seat.lower()
        if seat not in {"driver", "passenger"}:
            msg = f"未知座椅标识 {seat}(支持: driver, passenger)。"
            logger.warning("Invalid seat: %s", seat)
            return ToolResult(success=False, tool_name="set_seat_ventilation", output=msg, error="invalid_seat")
        if not (self._policy.seat_level_min <= level <= self._policy.seat_level_max):
            msg = (
                f"通风档位 {level} 超出范围 "
                f"[{self._policy.seat_level_min}, {self._policy.seat_level_max}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_seat_ventilation", output=msg, error="policy_violation")
        try:
            self._provider.set_seat_ventilation(seat, level)
            seat_cn = "驾驶位" if seat == "driver" else "副驾驶位"
            output = f"{seat_cn}通风已设置为 {level} 档。" if level > 0 else f"{seat_cn}通风已关闭。"
            logger.info("set_seat_ventilation(%s, %d) -> %s", seat, level, output)
            return ToolResult(
                success=True,
                tool_name="set_seat_ventilation",
                output=output,
                data={"seat": seat, "level": level},
            )
        except Exception as exc:
            logger.exception("set_seat_ventilation failed")
            return ToolResult(success=False, tool_name="set_seat_ventilation", output="", error=str(exc))

    def set_seat_massage(self, seat: str, level: int) -> ToolResult:
        """设置座椅按摩档位(0-3)。"""
        seat = seat.lower()
        if seat not in {"driver", "passenger"}:
            msg = f"未知座椅标识 {seat}(支持: driver, passenger)。"
            logger.warning("Invalid seat: %s", seat)
            return ToolResult(success=False, tool_name="set_seat_massage", output=msg, error="invalid_seat")
        if not (self._policy.seat_level_min <= level <= self._policy.seat_level_max):
            msg = (
                f"按摩档位 {level} 超出范围 "
                f"[{self._policy.seat_level_min}, {self._policy.seat_level_max}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_seat_massage", output=msg, error="policy_violation")
        try:
            self._provider.set_seat_massage(seat, level)
            seat_cn = "驾驶位" if seat == "driver" else "副驾驶位"
            output = f"{seat_cn}按摩已设置为 {level} 档。" if level > 0 else f"{seat_cn}按摩已关闭。"
            logger.info("set_seat_massage(%s, %d) -> %s", seat, level, output)
            return ToolResult(
                success=True,
                tool_name="set_seat_massage",
                output=output,
                data={"seat": seat, "level": level},
            )
        except Exception as exc:
            logger.exception("set_seat_massage failed")
            return ToolResult(success=False, tool_name="set_seat_massage", output="", error=str(exc))

    def set_seat_heating(self, seat: str, level: int) -> ToolResult:
        """设置座椅加热档位(0-3)。"""
        seat = seat.lower()
        if seat not in {"driver", "passenger"}:
            msg = f"未知座椅标识 {seat}(支持: driver, passenger)。"
            logger.warning("Invalid seat: %s", seat)
            return ToolResult(success=False, tool_name="set_seat_heating", output=msg, error="invalid_seat")
        if not (self._policy.seat_level_min <= level <= self._policy.seat_level_max):
            msg = (
                f"加热档位 {level} 超出范围 "
                f"[{self._policy.seat_level_min}, {self._policy.seat_level_max}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_seat_heating", output=msg, error="policy_violation")
        try:
            self._provider.set_seat_heating(seat, level)
            seat_cn = "驾驶位" if seat == "driver" else "副驾驶位"
            output = f"{seat_cn}加热已设置为 {level} 档。" if level > 0 else f"{seat_cn}加热已关闭。"
            logger.info("set_seat_heating(%s, %d) -> %s", seat, level, output)
            return ToolResult(
                success=True,
                tool_name="set_seat_heating",
                output=output,
                data={"seat": seat, "level": level},
            )
        except Exception as exc:
            logger.exception("set_seat_heating failed")
            return ToolResult(success=False, tool_name="set_seat_heating", output="", error=str(exc))

    # ---------------- 车窗 / 后备箱 ----------------

    def set_window(self, window: str, open_percent: int) -> ToolResult:
        """设置车窗开启百分比(车速过高时拒绝)。"""
        window = window.lower()
        valid_windows = {"all", "driver_front", "passenger_front", "driver_rear", "passenger_rear"}
        if window not in valid_windows:
            msg = f"未知车窗标识 {window}(支持: all, driver_front, passenger_front, driver_rear, passenger_rear)。"
            logger.warning("Invalid window: %s", window)
            return ToolResult(success=False, tool_name="set_window", output=msg, error="invalid_window")
        if not (
            self._policy.window_open_min_percent <= open_percent <= self._policy.window_open_max_percent
        ):
            msg = (
                f"车窗开启百分比 {open_percent} 超出范围 "
                f"[{self._policy.window_open_min_percent}, {self._policy.window_open_max_percent}]。"
            )
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_window", output=msg, error="policy_violation")
        try:
            state = self._provider.get_state()
            if state.speed_kmh > self._policy.max_speed_for_window_kmh and open_percent > 0:
                msg = (
                    f"当前车速 {state.speed_kmh:.0f} km/h 超过车窗操作安全阈值 "
                    f"{self._policy.max_speed_for_window_kmh:.0f} km/h,禁止开窗。"
                )
                logger.warning("Safety guard: %s", msg)
                return ToolResult(success=False, tool_name="set_window", output=msg, error="speed_too_high")
            self._provider.set_window(window, open_percent)
            window_cn_map = {
                "all": "所有车窗",
                "driver_front": "驾驶位前窗",
                "passenger_front": "副驾驶前窗",
                "driver_rear": "驾驶位后窗",
                "passenger_rear": "副驾驶后窗",
            }
            output = f"{window_cn_map[window]}已{'开启至 ' + str(open_percent) + '%' if open_percent > 0 else '完全关闭'}。"
            logger.info("set_window(%s, %d) -> %s", window, open_percent, output)
            return ToolResult(
                success=True,
                tool_name="set_window",
                output=output,
                data={"window": window, "open_percent": open_percent},
            )
        except Exception as exc:
            logger.exception("set_window failed")
            return ToolResult(success=False, tool_name="set_window", output="", error=str(exc))

    def set_trunk(self, open: bool) -> ToolResult:
        """开启或关闭后备箱(车速过高时拒绝)。"""
        try:
            state = self._provider.get_state()
            if state.speed_kmh > self._policy.max_speed_for_trunk_kmh:
                msg = (
                    f"当前车速 {state.speed_kmh:.0f} km/h 超过后备箱操作安全阈值 "
                    f"{self._policy.max_speed_for_trunk_kmh:.0f} km/h,禁止操作后备箱。"
                )
                logger.warning("Safety guard: %s", msg)
                return ToolResult(success=False, tool_name="set_trunk", output=msg, error="speed_too_high")
            self._provider.set_trunk(open)
            output = f"后备箱已{'开启' if open else '关闭'}。"
            logger.info("set_trunk(%s) -> %s", open, output)
            return ToolResult(
                success=True,
                tool_name="set_trunk",
                output=output,
                data={"trunk_open": open},
            )
        except Exception as exc:
            logger.exception("set_trunk failed")
            return ToolResult(success=False, tool_name="set_trunk", output="", error=str(exc))

    # ---------------- 雨刮 / 车灯 ----------------

    def set_wiper(self, level: str) -> ToolResult:
        """设置雨刮档位(off/low/medium/high/auto)。"""
        level = level.lower()
        valid_levels = {"off", "low", "medium", "high", "auto"}
        if level not in valid_levels:
            msg = f"未知雨刮档位 {level}(支持: off, low, medium, high, auto)。"
            logger.warning("Invalid wiper level: %s", level)
            return ToolResult(success=False, tool_name="set_wiper", output=msg, error="invalid_wiper_level")
        try:
            self._provider.set_wiper(level)
            level_cn_map = {
                "off": "关闭",
                "low": "低档",
                "medium": "中档",
                "high": "高档",
                "auto": "自动",
            }
            output = f"雨刮已设置为 {level_cn_map[level]}。"
            logger.info("set_wiper(%s) -> %s", level, output)
            return ToolResult(
                success=True,
                tool_name="set_wiper",
                output=output,
                data={"wiper_level": level},
            )
        except Exception as exc:
            logger.exception("set_wiper failed")
            return ToolResult(success=False, tool_name="set_wiper", output="", error=str(exc))

    def set_light(self, mode: str) -> ToolResult:
        """设置车灯模式(off/parking/low_beam/high_beam/auto)。"""
        mode = mode.lower()
        valid_modes = {"off", "parking", "low_beam", "high_beam", "auto"}
        if mode not in valid_modes:
            msg = f"未知车灯模式 {mode}(支持: off, parking, low_beam, high_beam, auto)。"
            logger.warning("Invalid light mode: %s", mode)
            return ToolResult(success=False, tool_name="set_light", output=msg, error="invalid_light_mode")
        try:
            self._provider.set_light(mode)
            mode_cn_map = {
                "off": "关闭",
                "parking": "示宽灯",
                "low_beam": "近光灯",
                "high_beam": "远光灯",
                "auto": "自动",
            }
            output = f"车灯已设置为 {mode_cn_map[mode]}。"
            logger.info("set_light(%s) -> %s", mode, output)
            return ToolResult(
                success=True,
                tool_name="set_light",
                output=output,
                data={"light_mode": mode},
            )
        except Exception as exc:
            logger.exception("set_light failed")
            return ToolResult(success=False, tool_name="set_light", output="", error=str(exc))


VEHICLE_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_vehicle_status",
        "description": "查询当前车辆状态,包括车速、电量、车内温度、空调状态、位置、座椅、车窗、后备箱、雨刮、车灯等。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_cabin_temperature",
        "description": "查询当前车内温度。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_temperature",
        "description": "设置目标车内温度(摄氏度,范围 18-30)。",
        "input_schema": {
            "type": "object",
            "properties": {"temperature_c": {"type": "number"}},
            "required": ["temperature_c"],
        },
    },
    {
        "name": "set_ac",
        "description": "开启或关闭空调。",
        "input_schema": {
            "type": "object",
            "properties": {"enabled": {"type": "boolean"}},
            "required": ["enabled"],
        },
    },
    {
        "name": "set_seat_position",
        "description": "设置座椅位置(seat: driver/passenger; slide_percent: 0-100前后滑动百分比; backrest_angle_deg: 0-180靠背角度)。至少指定一个参数。",
        "input_schema": {
            "type": "object",
            "properties": {
                "seat": {"type": "string", "enum": ["driver", "passenger"]},
                "slide_percent": {"type": "integer", "minimum": 0, "maximum": 100},
                "backrest_angle_deg": {"type": "integer", "minimum": 0, "maximum": 180},
            },
            "required": ["seat"],
        },
    },
    {
        "name": "set_seat_ventilation",
        "description": "设置座椅通风档位(seat: driver/passenger; level: 0=关闭,1-3=档位)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "seat": {"type": "string", "enum": ["driver", "passenger"]},
                "level": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": ["seat", "level"],
        },
    },
    {
        "name": "set_seat_massage",
        "description": "设置座椅按摩档位(seat: driver/passenger; level: 0=关闭,1-3=档位)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "seat": {"type": "string", "enum": ["driver", "passenger"]},
                "level": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": ["seat", "level"],
        },
    },
    {
        "name": "set_seat_heating",
        "description": "设置座椅加热档位(seat: driver/passenger; level: 0=关闭,1-3=档位)。",
        "input_schema": {
            "type": "object",
            "properties": {
                "seat": {"type": "string", "enum": ["driver", "passenger"]},
                "level": {"type": "integer", "minimum": 0, "maximum": 3},
            },
            "required": ["seat", "level"],
        },
    },
    {
        "name": "set_window",
        "description": "设置车窗开启百分比(window: all/driver_front/passenger_front/driver_rear/passenger_rear; open_percent: 0=全关,100=全开)。车速过高时禁止开窗。",
        "input_schema": {
            "type": "object",
            "properties": {
                "window": {"type": "string", "enum": ["all", "driver_front", "passenger_front", "driver_rear", "passenger_rear"]},
                "open_percent": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["window", "open_percent"],
        },
    },
    {
        "name": "set_trunk",
        "description": "开启或关闭后备箱(open: true=开启,false=关闭)。车速过高时禁止操作。",
        "input_schema": {
            "type": "object",
            "properties": {"open": {"type": "boolean"}},
            "required": ["open"],
        },
    },
    {
        "name": "set_wiper",
        "description": "设置雨刮档位(level: off/low/medium/high/auto)。",
        "input_schema": {
            "type": "object",
            "properties": {"level": {"type": "string", "enum": ["off", "low", "medium", "high", "auto"]}},
            "required": ["level"],
        },
    },
    {
        "name": "set_light",
        "description": "设置车灯模式(mode: off/parking/low_beam/high_beam/auto)。",
        "input_schema": {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["off", "parking", "low_beam", "high_beam", "auto"]}},
            "required": ["mode"],
        },
    },
]


def execute_vehicle_tool(name: str, arguments: dict, service: VehicleService) -> ToolResult:
    """根据工具名分派到 VehicleService 对应方法。"""
    if name == "get_vehicle_status":
        return service.get_vehicle_status()
    if name == "get_cabin_temperature":
        return service.get_cabin_temperature()
    if name == "set_temperature":
        return service.set_temperature(float(arguments.get("temperature_c", 0)))
    if name == "set_ac":
        return service.set_ac(bool(arguments.get("enabled", False)))
    if name == "set_seat_position":
        slide = arguments.get("slide_percent")
        backrest = arguments.get("backrest_angle_deg")
        return service.set_seat_position(
            str(arguments.get("seat", "")),
            int(slide) if slide is not None else None,
            int(backrest) if backrest is not None else None,
        )
    if name == "set_seat_ventilation":
        return service.set_seat_ventilation(
            str(arguments.get("seat", "")),
            int(arguments.get("level", 0)),
        )
    if name == "set_seat_massage":
        return service.set_seat_massage(
            str(arguments.get("seat", "")),
            int(arguments.get("level", 0)),
        )
    if name == "set_seat_heating":
        return service.set_seat_heating(
            str(arguments.get("seat", "")),
            int(arguments.get("level", 0)),
        )
    if name == "set_window":
        return service.set_window(
            str(arguments.get("window", "")),
            int(arguments.get("open_percent", 0)),
        )
    if name == "set_trunk":
        return service.set_trunk(bool(arguments.get("open", False)))
    if name == "set_wiper":
        return service.set_wiper(str(arguments.get("level", "")))
    if name == "set_light":
        return service.set_light(str(arguments.get("mode", "")))
    return ToolResult(success=False, tool_name=name, output="", error=f"unknown_tool:{name}")


_default_service: VehicleService | None = None


def get_vehicle_service() -> VehicleService:
    """获取默认 VehicleService 单例(场景池驱动,随机选景 + 会话级覆盖 + 偏好自动同步)。"""
    global _default_service
    if _default_service is not None:
        return _default_service
    policy = get_app_config().policy
    provider = SceneVehicleProvider(get_scene_pool())
    _default_service = VehicleService(provider, policy)
    return _default_service
