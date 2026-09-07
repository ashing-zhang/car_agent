# 配置加载模块 - 配置驱动,配置与代码分离
# 运行指南:
#   .env 存放敏感信息(API key, 数据库 URL),由 Pydantic Settings 加载
#   configs/*.yaml 存放非敏感运行配置,由数据模型加载
#   from app.config import get_settings, get_app_config

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class Settings(BaseSettings):
    """从 .env 加载的敏感配置(环境变量)。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_reasoning_model: str = "qwen-plus"

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"

    # Database
    postgres_url: str = "postgresql+psycopg2://autoagent:autoagent@localhost:5432/autoagent"
    redis_url: str = "redis://localhost:6379/0"

    # MCP
    mcp_host: str = "localhost"
    mcp_port: int = 8765

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    environment: str = "development"

    def resolved_llm_base_url(self) -> str:
        """返回实际使用的 LLM 端点,优先 llm_base_url,否则回退到 dashscope_base_url。"""
        return self.llm_base_url or self.dashscope_base_url

    def resolved_llm_api_key(self) -> str:
        """返回实际使用的 LLM API key。"""
        return self.llm_api_key or self.dashscope_api_key

    def resolved_llm_model(self) -> str:
        """返回实际使用的推理模型名。"""
        return self.llm_model or self.qwen_reasoning_model


class LLMConfig(BaseModel):
    """LLM 调用参数。"""

    temperature: float = 0.3
    max_tokens: int = 2048
    request_timeout: int = 30


class AgentConfig(BaseModel):
    """Agent 运行参数。"""

    max_iterations: int = 10
    max_tool_calls: int = 8
    reflection_enabled: bool = True


class MemoryConfig(BaseModel):
    """记忆系统参数。"""

    retrieval_top_k: int = 5
    min_confidence: float = 0.5
    conflict_resolution: str = "latest_wins"
    repository_backend: str = "in_memory"


class PolicyConfig(BaseModel):
    """工具安全策略参数。"""

    temperature_min: float = 18
    temperature_max: float = 30
    volume_min: int = 0
    volume_max: int = 40
    safety_critical_disabled: bool = True


class ObservabilityConfig(BaseModel):
    """可观测性参数(规格第20节,Phase 8)。"""

    trace_enabled: bool = True
    metrics_enabled: bool = True
    structured_logging: bool = True
    trace_store_limit: int = 200
    latency_buckets_ms: list[float] = [50, 100, 250, 500, 1000, 2500, 5000, 10000]


class AppYamlConfig(BaseModel):
    """从 configs/dev.yaml 加载的非敏感运行配置。"""

    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    memory: MemoryConfig = MemoryConfig()
    policy: PolicyConfig = PolicyConfig()
    observability: ObservabilityConfig = ObservabilityConfig()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取 .env 环境配置单例。"""
    return Settings()


def load_yaml_config(filename: str = "dev.yaml") -> dict:
    """加载指定 yaml 配置文件为字典。"""
    path = CONFIGS_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


@lru_cache(maxsize=1)
def get_app_config() -> AppYamlConfig:
    """获取应用 yaml 配置单例。"""
    return AppYamlConfig(**load_yaml_config("dev.yaml"))
