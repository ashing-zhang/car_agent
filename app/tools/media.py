# Media Tools - 媒体播放工具(规格第9节 media)
# 运行指南:
#   分层: Agent Tool → MediaService → MediaProvider(委托) → 媒体后端
#   默认 SimulatedMediaProvider,不依赖外部服务(Rule 5)

import logging
from typing import Protocol

from app.config import PolicyConfig, get_app_config
from app.tools.schemas import ToolResult

logger = logging.getLogger(__name__)


class MediaProvider(Protocol):
    """媒体播放能力提供者接口(委托模式)。"""

    def play(self, playlist: str) -> bool:
        """播放指定播放列表。"""
        ...

    def pause(self) -> None:
        """暂停播放。"""
        ...

    def set_volume(self, level: int) -> None:
        """设置音量。"""
        ...

    def get_status(self) -> dict:
        """返回当前播放状态。"""
        ...


class SimulatedMediaProvider:
    """模拟媒体播放提供者,内存维护播放状态。"""

    def __init__(self) -> None:
        """初始化停止状态。"""
        self._playing: bool = False
        self._playlist: str | None = None
        self._volume: int = 15

    def play(self, playlist: str) -> bool:
        """开始播放指定播放列表。"""
        self._playing = True
        self._playlist = playlist or "默认歌单"
        logger.info("Media playing: %s", self._playlist)
        return True

    def pause(self) -> None:
        """暂停播放。"""
        self._playing = False
        logger.info("Media paused")

    def set_volume(self, level: int) -> None:
        """设置音量。"""
        self._volume = level
        logger.info("Media volume set to %d", level)

    def get_status(self) -> dict:
        """返回当前播放状态。"""
        return {
            "playing": self._playing,
            "playlist": self._playlist,
            "volume": self._volume,
        }


class MediaService:
    """媒体服务层,policy 校验后委托给 MediaProvider。"""

    def __init__(self, provider: MediaProvider, policy: PolicyConfig) -> None:
        """注入媒体提供者与安全策略。"""
        self._provider = provider
        self._policy = policy

    def play_media(self, playlist: str) -> ToolResult:
        """播放指定播放列表。"""
        self._provider.play(playlist)
        status = self._provider.get_status()
        output = f"正在播放「{status['playlist']}」。"
        return ToolResult(
            success=True,
            tool_name="play_media",
            output=output,
            data={"playlist": status["playlist"]},
        )

    def pause_media(self) -> ToolResult:
        """暂停播放。"""
        self._provider.pause()
        return ToolResult(success=True, tool_name="pause_media", output="已暂停播放。")

    def set_volume(self, level: int) -> ToolResult:
        """设置音量(范围 0-40)。"""
        if not (self._policy.volume_min <= level <= self._policy.volume_max):
            msg = f"音量 {level} 超出安全范围 [{self._policy.volume_min}, {self._policy.volume_max}]。"
            logger.warning("Policy violation: %s", msg)
            return ToolResult(success=False, tool_name="set_volume", output=msg, error="policy_violation")
        self._provider.set_volume(level)
        output = f"音量已设置为 {level}。"
        return ToolResult(success=True, tool_name="set_volume", output=output, data={"volume": level})


MEDIA_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "play_media",
        "description": "播放指定播放列表或歌单。",
        "input_schema": {
            "type": "object",
            "properties": {"playlist": {"type": "string"}},
            "required": ["playlist"],
        },
    },
    {
        "name": "pause_media",
        "description": "暂停当前媒体播放。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_volume",
        "description": "设置媒体音量(范围 0-40)。",
        "input_schema": {
            "type": "object",
            "properties": {"level": {"type": "integer"}},
            "required": ["level"],
        },
    },
]


def execute_media_tool(name: str, arguments: dict, service: MediaService) -> ToolResult:
    """分派到 MediaService 对应方法。"""
    if name == "play_media":
        return service.play_media(arguments.get("playlist", ""))
    if name == "pause_media":
        return service.pause_media()
    if name == "set_volume":
        return service.set_volume(int(arguments.get("level", 0)))
    return ToolResult(success=False, tool_name=name, output="", error=f"unknown_tool:{name}")


_default_service: MediaService | None = None


def get_media_service() -> MediaService:
    """获取默认 MediaService 单例。"""
    global _default_service
    if _default_service is None:
        policy = get_app_config().policy
        _default_service = MediaService(SimulatedMediaProvider(), policy)
    return _default_service
