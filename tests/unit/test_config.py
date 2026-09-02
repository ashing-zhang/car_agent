# 配置加载单元测试
# 运行指南: pytest tests/unit/test_config.py -v

from app.config import AppYamlConfig, get_app_config, get_settings


def test_settings_loaded() -> None:
    """验证 .env 环境配置可被加载。"""
    settings = get_settings()
    assert settings.environment == "development"


def test_yaml_config_loaded() -> None:
    """验证 yaml 配置可被加载。"""
    config = get_app_config()
    assert isinstance(config, AppYamlConfig)
    assert config.policy.temperature_min == 18
    assert config.policy.temperature_max == 30


def test_llm_endpoint_resolution() -> None:
    """验证 LLM 端点解析逻辑。"""
    settings = get_settings()
    base_url = settings.resolved_llm_base_url()
    assert base_url != ""
