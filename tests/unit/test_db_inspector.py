# 数据库探查工具单元测试 - 覆盖配置解析、schema 过滤、值序列化与报告输出
# 运行指南:
#   python -m pytest tests/unit/test_db_inspector.py -v
#   (不依赖真实 Postgres,使用 unittest.mock 模拟 SQLAlchemy 组件)

import io
import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tools.db_inspector import (
    SYSTEM_SCHEMAS,
    ColumnInfo,
    ConsoleReporter,
    ConstraintInfo,
    DatabaseInspector,
    DatabaseSnapshot,
    DbInspectorConfig,
    EngineFactory,
    IndexInfo,
    InspectionConfig,
    JsonReporter,
    OutputConfig,
    TableInfo,
    load_inspector_config,
)

# ---------------------------------------------------------------------------
# 配置模型测试
# ---------------------------------------------------------------------------

class TestConfigModels:
    """Pydantic 配置模型验证。"""

    def test_default_output_config(self) -> None:
        """默认输出配置值符合预期。"""
        cfg = OutputConfig()
        assert cfg.format == "both"
        assert cfg.json_output_dir == "reports/db_inspect"
        assert cfg.max_rows_per_table_console == 50
        assert cfg.max_rows_per_table_json is None

    def test_output_format_enforced(self) -> None:
        """format 字段只接受枚举内值。"""
        from pydantic import ValidationError

        for valid in ("table", "json", "both"):
            assert OutputConfig(format=valid).format == valid
        with pytest.raises(ValidationError):
            OutputConfig(format="xml")

    def test_inspection_defaults(self) -> None:
        """探查范围默认值。"""
        cfg = InspectionConfig()
        assert cfg.include_system_schemas is False
        assert cfg.schemas == []
        assert cfg.tables == []
        assert cfg.include_table_data is True
        assert cfg.include_indexes is True
        assert cfg.include_constraints is True
        assert cfg.truncate_large_fields is True
        assert cfg.truncate_threshold == 200

    def test_db_inspector_config_top_level_defaults(self) -> None:
        """顶层配置三个子对象默认存在。"""
        cfg = DbInspectorConfig()
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.inspection, InspectionConfig)
        assert cfg.connection.connect_timeout == 10

    def test_load_inspector_config_returns_default_when_missing(self) -> None:
        """不存在的配置文件名时返回默认值。"""
        cfg = load_inspector_config("nonexistent_file_xyz.yaml")
        assert isinstance(cfg, DbInspectorConfig)
        assert cfg.output.format == "both"


# ---------------------------------------------------------------------------
# 结果数据模型测试
# ---------------------------------------------------------------------------

class TestSnapshotModels:
    """数据模型序列化与默认值。"""

    def test_column_info_defaults(self) -> None:
        """ColumnInfo 默认值。"""
        c = ColumnInfo(name="id", type="integer", nullable=False)
        assert c.default is None
        assert c.comment is None
        assert c.is_primary_key is False

    def test_table_info_roundtrip_json(self) -> None:
        """TableInfo 能 JSON 序列化再还原。"""
        t = TableInfo(
            schema="public",
            name="users",
            comment="用户表",
            row_count=2,
            columns=[ColumnInfo(name="id", type="int", nullable=False, is_primary_key=True)],
            indexes=[IndexInfo(name="pk_users", columns=["id"], is_primary=True)],
            constraints=[ConstraintInfo(name="pk_users", type="PRIMARY KEY", columns=["id"])],
            rows=[{"id": 1}, {"id": 2}],
        )
        raw = t.model_dump_json()
        restored = TableInfo.model_validate_json(raw)
        assert restored.name == "users"
        assert restored.row_count == 2
        assert restored.columns[0].name == "id"
        assert restored.rows[0]["id"] == 1

    def test_database_snapshot_jsonable(self) -> None:
        """DatabaseSnapshot model_dump_json 不抛异常。"""
        snap = DatabaseSnapshot(
            generated_at=datetime.now(UTC).isoformat(),
            database_name="testdb",
            schemas=["public"],
            tables=[],
        )
        obj = json.loads(snap.model_dump_json())
        assert obj["database_name"] == "testdb"
        assert obj["schemas"] == ["public"]
        assert "generated_at" in obj


# ---------------------------------------------------------------------------
# EngineFactory 测试
# ---------------------------------------------------------------------------

class TestEngineFactory:
    """EngineFactory 行为验证。"""

    @patch("app.tools.db_inspector.create_engine")
    def test_build_succeeds_on_first_try(self, mock_create: MagicMock) -> None:
        """首次连接成功时不重试。"""
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value.execute.return_value = None
        mock_create.return_value = engine

        cfg = DbInspectorConfig()
        factory = EngineFactory(cfg)
        got = factory.build()

        assert got is engine
        assert mock_create.call_count == 1

    @patch("app.tools.db_inspector.time.sleep", return_value=None)
    @patch("app.tools.db_inspector.create_engine")
    def test_build_retries_on_operational_error(
        self, mock_create: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """遇到 OperationalError 时最多重试 3 次。"""
        from sqlalchemy.exc import OperationalError

        bad_engine = MagicMock()
        bad_engine.connect.side_effect = OperationalError("m", {}, BaseException("boom"))
        good_engine = MagicMock()
        good_engine.connect.return_value.__enter__.return_value.execute.return_value = None
        mock_create.side_effect = [bad_engine, bad_engine, good_engine]

        cfg = DbInspectorConfig()
        factory = EngineFactory(cfg)
        got = factory.build()

        assert got is good_engine
        assert mock_create.call_count == 3
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# DatabaseInspector 逻辑测试 (mock SQLAlchemy Inspector)
# ---------------------------------------------------------------------------

class _FakeInspector:
    """最小 mock 版 SQLAlchemy Inspector,用于测试 _filter_schemas 等纯逻辑路径。"""

    def __init__(self, schemas: list[str], tables_by_schema: dict[str, list[dict]]) -> None:
        """保存预置模拟数据。"""
        self._schemas = schemas
        self._tables = tables_by_schema

    def get_schema_names(self) -> list[str]:
        return list(self._schemas)

    def get_table_names(self, schema: str = "public") -> list[str]:
        return [t["name"] for t in self._tables.get(schema, [])]

    def get_columns(self, table_name: str, schema: str = "public") -> list[dict]:
        for t in self._tables.get(schema, []):
            if t["name"] == table_name:
                return t["columns"]
        return []

    def get_pk_constraint(self, table_name: str, schema: str = "public") -> dict:
        for t in self._tables.get(schema, []):
            if t["name"] == table_name:
                return t.get("pk", {"constrained_columns": [], "name": None})
        return {"constrained_columns": [], "name": None}

    def get_indexes(self, table_name: str, schema: str = "public") -> list[dict]:
        for t in self._tables.get(schema, []):
            if t["name"] == table_name:
                return t.get("indexes", [])
        return []

    def get_foreign_keys(self, table_name: str, schema: str = "public") -> list[dict]:
        return []

    def get_unique_constraints(self, table_name: str, schema: str = "public") -> list[dict]:
        return []

    def get_check_constraints(self, table_name: str, schema: str = "public") -> list[dict]:
        return []

    def get_table_comment(self, table_name: str, schema: str = "public") -> dict:
        for t in self._tables.get(schema, []):
            if t["name"] == table_name:
                return {"text": t.get("comment")}
        return {"text": None}


class TestDatabaseInspectorLogic:
    """DatabaseInspector 的纯逻辑方法 (mock 后端)。"""

    @staticmethod
    def _build_inspector(
        config: DbInspectorConfig, inspector: "_FakeInspector"
    ) -> DatabaseInspector:
        """构造已初始化(绕过真实连接)的 DatabaseInspector。"""
        factory = MagicMock(spec=EngineFactory)
        insp = DatabaseInspector(config, engine_factory=factory)
        insp._engine = MagicMock()
        insp._inspector = inspector
        return insp

    def test_filter_schemas_excludes_system(self) -> None:
        """默认排除 pg_catalog 等系统 schema。"""
        all_schemas = ["public", "pg_catalog", "information_schema", "custom"]
        insp = self._build_inspector(DbInspectorConfig(), _FakeInspector([], {}))
        result = insp._filter_schemas(all_schemas, InspectionConfig())
        for s in SYSTEM_SCHEMAS:
            assert s not in result
        assert "public" in result and "custom" in result

    def test_filter_schemas_include_system_flag(self) -> None:
        """打开 include_system_schemas 时系统 schema 保留。"""
        all_schemas = ["public", "pg_catalog"]
        insp = self._build_inspector(DbInspectorConfig(), _FakeInspector([], {}))
        cfg = InspectionConfig(include_system_schemas=True)
        result = insp._filter_schemas(all_schemas, cfg)
        assert "pg_catalog" in result

    def test_filter_schemas_whitelist(self) -> None:
        """schemas 白名单生效。"""
        all_schemas = ["public", "audit", "custom"]
        insp = self._build_inspector(DbInspectorConfig(), _FakeInspector([], {}))
        cfg = InspectionConfig(schemas=["public", "audit"])
        assert insp._filter_schemas(all_schemas, cfg) == ["public", "audit"]

    def test_sanitize_value_handles_datetime(self) -> None:
        """日期时间类型转为 ISO 字符串。"""
        cfg = InspectionConfig()
        insp = self._build_inspector(DbInspectorConfig(), _FakeInspector([], {}))
        dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
        assert insp._sanitize_value(dt, cfg) == "2024-01-02T03:04:05+00:00"
        assert insp._sanitize_value(date(2024, 1, 2), cfg) == "2024-01-02"

    def test_sanitize_value_truncates_long_string(self) -> None:
        """长文本按配置截断并提示原长度。"""
        cfg = InspectionConfig(truncate_large_fields=True, truncate_threshold=10)
        insp = self._build_inspector(DbInspectorConfig(), _FakeInspector([], {}))
        s = "a" * 100
        out = insp._sanitize_value(s, cfg)
        assert isinstance(out, str)
        assert out.startswith("a" * 10)
        assert "100 chars" in out

    def test_sanitize_value_preserves_primitives(self) -> None:
        """基础类型原样保留。"""
        cfg = InspectionConfig()
        insp = self._build_inspector(DbInspectorConfig(), _FakeInspector([], {}))
        assert insp._sanitize_value(None, cfg) is None
        assert insp._sanitize_value(True, cfg) is True
        assert insp._sanitize_value(42, cfg) == 42
        assert insp._sanitize_value(3.14, cfg) == 3.14
        assert insp._sanitize_value([1, 2], cfg) == [1, 2]
        assert insp._sanitize_value({"k": "v"}, cfg) == {"k": "v"}

    def test_inspect_full_flow_mocked(self) -> None:
        """完整 inspect 流程使用 mock Inspector 验证编排逻辑。"""
        fake_inspector = _FakeInspector(
            schemas=["public"],
            tables_by_schema={
                "public": [
                    {
                        "name": "users",
                        "comment": "用户表",
                        "pk": {"name": "pk_users", "constrained_columns": ["id"]},
                        "columns": [
                            {"name": "id", "type": "INTEGER", "nullable": False, "default": None},
                            {"name": "name", "type": "VARCHAR(64)", "nullable": False, "default": None},
                        ],
                        "indexes": [
                            {"name": "pk_users", "column_names": ["id"], "unique": True},
                        ],
                    },
                ],
            },
        )
        config = DbInspectorConfig(
            inspection=InspectionConfig(include_table_data=False),
            output=OutputConfig(format="table"),
        )
        insp = self._build_inspector(config, fake_inspector)
        # mock _engine.connect / current_database
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = ["autoagent"]
        insp._engine.connect.return_value.__enter__.return_value = mock_conn

        snapshot = insp.inspect()

        assert snapshot.database_name == "autoagent"
        assert snapshot.schemas == ["public"]
        assert len(snapshot.tables) == 1
        t = snapshot.tables[0]
        assert t.schema == "public" and t.name == "users"
        assert t.comment == "用户表"
        assert len(t.columns) == 2
        assert t.columns[0].is_primary_key is True
        assert len(t.indexes) == 1
        assert t.indexes[0].unique is True

    def test_context_manager_lifecycle(self) -> None:
        """with 语句下 engine 被创建与释放。"""
        factory = MagicMock(spec=EngineFactory)
        engine = MagicMock()
        factory.build.return_value = engine

        insp = DatabaseInspector(DbInspectorConfig(), engine_factory=factory)
        with patch("app.tools.db_inspector.inspect") as mock_inspect_fn:
            mock_inspect_fn.return_value = MagicMock()
            with insp as ctx:
                assert ctx is insp
                assert factory.build.call_count == 1
                assert engine.dispose.call_count == 0
        assert engine.dispose.call_count == 1
        assert insp._engine is None
        assert insp._inspector is None


# ---------------------------------------------------------------------------
# Reporter 测试
# ---------------------------------------------------------------------------

class TestReporters:
    """ConsoleReporter 与 JsonReporter 输出验证。"""

    @staticmethod
    def _sample_snapshot() -> DatabaseSnapshot:
        """构造小型样例快照。"""
        return DatabaseSnapshot(
            generated_at=datetime(2024, 1, 1, tzinfo=UTC).isoformat(),
            database_name="sample",
            schemas=["public"],
            tables=[
                TableInfo(
                    schema="public",
                    name="t1",
                    comment="demo",
                    row_count=2,
                    columns=[
                        ColumnInfo(name="id", type="INT", nullable=False, is_primary_key=True),
                        ColumnInfo(name="val", type="TEXT", nullable=True),
                    ],
                    indexes=[IndexInfo(name="pk_t1", columns=["id"], unique=True, is_primary=True)],
                    constraints=[
                        ConstraintInfo(name="pk_t1", type="PRIMARY KEY", columns=["id"]),
                    ],
                    rows=[{"id": 1, "val": "a"}, {"id": 2, "val": "b"}],
                ),
            ],
        )

    def test_console_reporter_prints_tables(self) -> None:
        """ConsoleReporter 输出含有关键段标题。"""
        cfg = DbInspectorConfig(
            output=OutputConfig(format="table", max_rows_per_table_console=10),
            inspection=InspectionConfig(),
        )
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            ConsoleReporter(cfg).print(self._sample_snapshot())
        out = buf.getvalue()
        assert "sample" in out
        assert "Table: public.t1" in out
        assert "Columns" in out
        assert "Indexes" in out
        assert "Constraints" in out
        assert "Rows" in out

    def test_json_reporter_writes_file(self, tmp_path: Path) -> None:
        """JsonReporter 写出 JSON 文件,内容可解析。"""
        cfg = DbInspectorConfig(
            output=OutputConfig(json_output_dir=str(tmp_path.relative_to(Path.cwd())))
        )
        # 由于 JsonReporter 使用 PROJECT_ROOT,我们 patch 它
        with patch("app.tools.db_inspector.PROJECT_ROOT", Path.cwd()):
            path = JsonReporter(cfg).write(self._sample_snapshot())
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["database_name"] == "sample"
        assert data["tables"][0]["name"] == "t1"
