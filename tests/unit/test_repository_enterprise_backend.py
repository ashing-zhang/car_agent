# Repository 企业级后端验证测试 - 验证默认仓储统一使用 PostgreSQLMemoryRepository
# 运行指南: pytest tests/unit/test_repository_enterprise_backend.py -v

from unittest.mock import patch, MagicMock

from app.evaluation.evaluator import Evaluator
from app.memory.repository import PostgreSQLMemoryRepository, build_repository, get_memory_repository, reset_memory_repository


def test_build_repository_returns_pg_instance() -> None:
    """验证 build_repository() 返回 PostgreSQLMemoryRepository 实例(企业级后端)。"""
    with patch("app.memory.repository.PostgreSQLMemoryRepository.__init__", return_value=None):
        repo = build_repository()
        assert isinstance(repo, PostgreSQLMemoryRepository)
        assert repo.backend == "postgres"


def test_get_memory_repository_returns_pg_singleton() -> None:
    """验证 get_memory_repository() 默认返回 PostgreSQL 单例。"""
    reset_memory_repository()
    with patch("app.memory.repository.PostgreSQLMemoryRepository.__init__", return_value=None):
        repo1 = get_memory_repository()
        repo2 = get_memory_repository()
        assert repo1 is repo2
        assert isinstance(repo1, PostgreSQLMemoryRepository)


def test_evaluator_react_agent_uses_pg_repository() -> None:
    """验证 Evaluator._make_react_agent 使用 build_repository() 即 PostgreSQL 仓储。"""
    with patch("app.evaluation.evaluator.build_repository") as mock_build:
        mock_repo = MagicMock()
        mock_build.return_value = mock_repo
        with patch("app.evaluation.evaluator.build_agent") as mock_build_agent, \
             patch.object(Evaluator, "_make_shared_services", return_value=(MagicMock(), MagicMock())):
            evaluator = Evaluator()
            evaluator._make_react_agent()
            mock_build.assert_called_once_with()


def test_pg_repository_backend_attribute() -> None:
    """验证 PostgreSQLMemoryRepository 的 backend 标识为 'postgres'。"""
    with patch("app.memory.repository.PostgreSQLMemoryRepository.__init__", return_value=None):
        repo = PostgreSQLMemoryRepository()
        assert repo.backend == "postgres"
