# Observability 日志 - 结构化 JSON 日志(规格第20节,Phase 8)
# 运行指南:
#   from app.observability.logging import setup_logging
#   setup_logging(level="INFO", structured=True)
#   import logging; logging.getLogger("app").info("hello", extra={"request_id":"r-1"})

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """将 LogRecord 格式化为单行 JSON,带 trace_id 等额外字段。"""

    def format(self, record: logging.LogRecord) -> str:
        """输出 JSON 字符串。"""
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        extras = getattr(record, "__dict__", {})
        for key in ("request_id", "session_id", "user_id", "tool", "latency_ms"):
            if key in extras:
                payload[key] = extras[key]
        for key, value in extras.items():
            if key not in payload and key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """人类可读的控制台格式。"""

    DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

    def __init__(self, fmt: str | None = None) -> None:
        """初始化 formatter。"""
        super().__init__(fmt or self.DEFAULT_FMT)


def setup_logging(level: str = "INFO", structured: bool = True) -> None:
    """配置根 logger 的 handler 与格式(structured=True 用 JSON)。"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if structured else PlainFormatter())
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger。"""
    return logging.getLogger(name)
