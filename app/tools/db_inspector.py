# 数据库探查工具 - 探查 Postgres 容器中的所有表结构、字段及记录
# 运行指南:
#   前置条件: docker compose up -d postgres 启动 Postgres 容器
#   使用方式: 通过 scripts/inspect_db.py 调用 (python -m scripts.inspect_db)
#   或在代码中: from app.tools.db_inspector import DatabaseInspector, load_inspector_config

import json
import logging
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, Field
from sqlalchemy import Engine, MetaData, Table, create_engine, inspect, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from app.config import CONFIGS_DIR, PROJECT_ROOT, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 配置数据模型 (配置驱动)
# ---------------------------------------------------------------------------

class OutputConfig(BaseModel):
    """输出格式相关配置。"""

    format: str = Field(default="both", pattern="^(table|json|both)$")
    json_output_dir: str = "reports/db_inspect"
    max_rows_per_table_console: int = 50
    max_rows_per_table_json: int | None = None


class InspectionConfig(BaseModel):
    """探查范围与深度配置。"""

    include_system_schemas: bool = False
    schemas: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    include_table_data: bool = True
    include_indexes: bool = True
    include_constraints: bool = True
    truncate_large_fields: bool = True
    truncate_threshold: int = 200


class ConnectionConfig(BaseModel):
    """数据库连接参数(可选覆盖)。"""

    override_url: str = ""
    connect_timeout: int = 10
    statement_timeout: int = 300


class DbInspectorConfig(BaseModel):
    """数据库探查工具的顶层配置模型。"""

    output: OutputConfig = Field(default_factory=OutputConfig)
    inspection: InspectionConfig = Field(default_factory=InspectionConfig)
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)


# ---------------------------------------------------------------------------
# 探查结果数据模型
# ---------------------------------------------------------------------------

class ColumnInfo(BaseModel):
    """列信息。"""

    name: str
    type: str
    nullable: bool
    default: str | None = None
    comment: str | None = None
    is_primary_key: bool = False


class IndexInfo(BaseModel):
    """索引信息。"""

    name: str
    columns: list[str]
    unique: bool = False
    is_primary: bool = False


class ConstraintInfo(BaseModel):
    """约束信息。"""

    name: str
    type: str
    columns: list[str]
    referred_table: str | None = None
    referred_columns: list[str] | None = None


class TableInfo(BaseModel):
    """单张表的完整信息(结构+索引+约束+数据)。"""

    schema: str
    name: str
    comment: str | None = None
    row_count: int = 0
    columns: list[ColumnInfo] = Field(default_factory=list)
    indexes: list[IndexInfo] = Field(default_factory=list)
    constraints: list[ConstraintInfo] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)


class DatabaseSnapshot(BaseModel):
    """整个数据库的探查快照。"""

    generated_at: str
    database_name: str
    schemas: list[str]
    tables: list[TableInfo]


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_inspector_config(filename: str = "db_inspector.yaml") -> DbInspectorConfig:
    """从 configs/ 目录加载探查配置。

    Args:
        filename: 配置文件名,默认为 db_inspector.yaml

    Returns:
        解析后的 DbInspectorConfig 实例(文件不存在时返回默认值)
    """
    path = CONFIGS_DIR / filename
    if not path.exists():
        logger.warning("Config %s not found, using defaults", path)
        return DbInspectorConfig()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return DbInspectorConfig(**raw)


# ---------------------------------------------------------------------------
# 引擎工厂 (含重试)
# ---------------------------------------------------------------------------

class EngineFactory:
    """Engine 工厂,含连接重试机制。"""

    def __init__(self, config: DbInspectorConfig) -> None:
        """保存配置引用。"""
        self._config = config

    def build(self) -> Engine:
        """构建 SQLAlchemy Engine,含最多 3 次重试。

        Returns:
            已就绪的 Engine 实例

        Raises:
            OperationalError: 多次重试后仍无法连接
        """
        cfg = self._config.connection
        url = cfg.override_url or get_settings().postgres_url
        connect_args = {"connect_timeout": cfg.connect_timeout}
        max_retries = 3
        last_err: OperationalError | None = None
        for attempt in range(1, max_retries + 1):
            try:
                engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True, future=True)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                    if cfg.statement_timeout:
                        conn.execute(text(f"SET statement_timeout = {cfg.statement_timeout * 1000}"))
                logger.info("Connected to database successfully (attempt %d)", attempt)
                return engine
            except OperationalError as e:
                last_err = e
                logger.warning("Connection attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    time.sleep(2 * attempt)
        assert last_err is not None
        raise last_err


# ---------------------------------------------------------------------------
# 核心探查器 (委托模式,不继承任何类)
# ---------------------------------------------------------------------------

SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}


class DatabaseInspector:
    """数据库探查器 - 负责提取表结构、索引、约束和数据。

    通过委托 EngineFactory 构建连接、委托 SQLAlchemy Inspector 反射元数据,
    自身只负责编排业务逻辑与输出格式转换。
    """

    def __init__(
        self,
        config: DbInspectorConfig,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        """保存配置并注入依赖。

        Args:
            config: 探查配置
            engine_factory: 可选的 EngineFactory,未提供时使用默认实现
        """
        self._config = config
        self._engine_factory = engine_factory or EngineFactory(config)
        self._engine: Engine | None = None
        self._inspector: Inspector | None = None

    # ---- 生命周期 ---------------------------------------------------------

    def __enter__(self) -> Self:
        """进入上下文时建立连接。"""
        self._engine = self._engine_factory.build()
        self._inspector = inspect(self._engine)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """退出上下文时释放连接。"""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._inspector = None

    # ---- 公共 API ---------------------------------------------------------

    def inspect(self) -> DatabaseSnapshot:
        """执行完整的数据库探查并返回结构化快照。

        Returns:
            DatabaseSnapshot 包含所有被选中的 schema 与表信息
        """
        if self._inspector is None:
            raise RuntimeError("Inspector is not connected. Use 'with DatabaseInspector(...) as insp:'")
        insp_cfg = self._config.inspection

        all_schemas = self._inspector.get_schema_names()
        schemas = self._filter_schemas(all_schemas, insp_cfg)
        logger.info("Inspecting schemas: %s (system=%s)", schemas, insp_cfg.include_system_schemas)

        tables: list[TableInfo] = []
        for schema in schemas:
            table_names = self._inspector.get_table_names(schema=schema)
            if insp_cfg.tables:
                table_names = [t for t in table_names if t in set(insp_cfg.tables)]
            for tname in table_names:
                try:
                    table_info = self._inspect_table(schema, tname)
                    tables.append(table_info)
                    logger.info("  %s.%s: %d columns, %d rows", schema, tname,
                                len(table_info.columns), table_info.row_count)
                except SQLAlchemyError as e:
                    logger.error("Failed to inspect table %s.%s: %s", schema, tname, e)

        db_name = self._resolve_database_name()
        return DatabaseSnapshot(
            generated_at=datetime.now(UTC).isoformat(),
            database_name=db_name,
            schemas=schemas,
            tables=tables,
        )

    # ---- 内部辅助 ---------------------------------------------------------

    def _filter_schemas(self, all_schemas: list[str], cfg: InspectionConfig) -> list[str]:
        """根据配置过滤 schema。"""
        result = all_schemas
        if not cfg.include_system_schemas:
            result = [s for s in result if s not in SYSTEM_SCHEMAS]
        if cfg.schemas:
            allowed = set(cfg.schemas)
            result = [s for s in result if s in allowed]
        return result

    def _resolve_database_name(self) -> str:
        """从当前连接中提取数据库名。"""
        assert self._engine is not None
        try:
            with self._engine.connect() as conn:
                row = conn.execute(text("SELECT current_database()")).fetchone()
                return str(row[0]) if row else "unknown"
        except SQLAlchemyError:
            return "unknown"

    def _inspect_table(self, schema: str, table_name: str) -> TableInfo:
        """探查单张表的所有信息。"""
        assert self._inspector is not None and self._engine is not None
        insp_cfg = self._config.inspection

        columns_raw = self._inspector.get_columns(table_name, schema=schema)
        pk_cols = set(self._inspector.get_pk_constraint(table_name, schema=schema).get("constrained_columns", []))
        columns = [
            ColumnInfo(
                name=c["name"],
                type=str(c["type"]),
                nullable=bool(c.get("nullable", True)),
                default=str(c["default"]) if c.get("default") is not None else None,
                comment=c.get("comment"),
                is_primary_key=c["name"] in pk_cols,
            )
            for c in columns_raw
        ]

        indexes: list[IndexInfo] = []
        if insp_cfg.include_indexes:
            for idx in self._inspector.get_indexes(table_name, schema=schema):
                indexes.append(IndexInfo(
                    name=idx["name"],
                    columns=list(idx["column_names"]),
                    unique=bool(idx.get("unique", False)),
                    is_primary=idx["name"].startswith("pk_") or set(idx["column_names"]) == pk_cols,
                ))

        constraints: list[ConstraintInfo] = []
        if insp_cfg.include_constraints:
            constraints.extend(self._collect_constraints(table_name, schema))

        comment = self._inspector.get_table_comment(table_name, schema=schema).get("text")
        row_count = self._estimate_row_count(schema, table_name)

        rows: list[dict[str, Any]] = []
        if insp_cfg.include_table_data:
            rows = self._fetch_rows(schema, table_name, columns)
            row_count = len(rows) if rows else row_count

        return TableInfo(
            schema=schema,
            name=table_name,
            comment=comment,
            row_count=row_count,
            columns=columns,
            indexes=indexes,
            constraints=constraints,
            rows=rows,
        )

    def _collect_constraints(self, table_name: str, schema: str) -> list[ConstraintInfo]:
        """收集主键、外键、唯一、检查约束。"""
        assert self._inspector is not None
        result: list[ConstraintInfo] = []

        pk = self._inspector.get_pk_constraint(table_name, schema=schema)
        if pk and pk.get("constrained_columns"):
            result.append(ConstraintInfo(
                name=pk.get("name", f"pk_{table_name}"),
                type="PRIMARY KEY",
                columns=list(pk["constrained_columns"]),
            ))

        for fk in self._inspector.get_foreign_keys(table_name, schema=schema):
            result.append(ConstraintInfo(
                name=fk.get("name", f"fk_{table_name}"),
                type="FOREIGN KEY",
                columns=list(fk.get("constrained_columns", [])),
                referred_table=fk.get("referred_table"),
                referred_columns=list(fk.get("referred_columns", [])) if fk.get("referred_columns") else None,
            ))

        for uc in self._inspector.get_unique_constraints(table_name, schema=schema):
            result.append(ConstraintInfo(
                name=uc.get("name", f"uq_{table_name}"),
                type="UNIQUE",
                columns=list(uc.get("column_names", [])),
            ))

        for cc in self._inspector.get_check_constraints(table_name, schema=schema):
            result.append(ConstraintInfo(
                name=cc.get("name", f"ck_{table_name}"),
                type="CHECK",
                columns=list(cc.get("column_names", [])),
            ))

        return result

    def _estimate_row_count(self, schema: str, table_name: str) -> int:
        """通过 pg_class.reltuples 快速估算行数,失败时返回 0。"""
        assert self._engine is not None
        try:
            with self._engine.connect() as conn:
                q = text("""
                    SELECT reltuples::bigint
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = :schema AND c.relname = :table
                """)
                row = conn.execute(q, {"schema": schema, "table": table_name}).fetchone()
                return int(row[0]) if row and row[0] is not None else 0
        except SQLAlchemyError as e:
            logger.debug("Row count estimate failed for %s.%s: %s", schema, table_name, e)
            return 0

    def _fetch_rows(
        self, schema: str, table_name: str, columns: list[ColumnInfo]
    ) -> list[dict[str, Any]]:
        """从指定表读取记录并做字段截断/序列化处理。

        JSON 导出与控制台共用同一逻辑,只是 max_rows 上限不同由上层控制。
        """
        assert self._engine is not None
        max_rows = (
            self._config.output.max_rows_per_table_console
            if self._config.output.format == "table"
            else (self._config.output.max_rows_per_table_json)
        )

        metadata = MetaData(schema=schema)
        try:
            table = Table(table_name, metadata, autoload_with=self._engine)
        except SQLAlchemyError as e:
            logger.error("Cannot reflect table %s.%s: %s", schema, table_name, e)
            return []

        try:
            with self._engine.connect() as conn:
                stmt = table.select()
                if max_rows is not None:
                    stmt = stmt.limit(max_rows)
                cursor_result = conn.execute(stmt)
                rows_raw = cursor_result.mappings().all()
        except SQLAlchemyError as e:
            logger.error("Select from %s.%s failed: %s", schema, table_name, e)
            return []

        col_names = {c.name for c in columns}
        return [self._sanitize_row(dict(r), col_names) for r in rows_raw]

    def _sanitize_row(self, row: dict[str, Any], col_names: set[str]) -> dict[str, Any]:
        """将一行数据做 JSON 可序列化转换,按需截断大字段。"""
        out: dict[str, Any] = {}
        insp_cfg = self._config.inspection
        for key, val in row.items():
            if key not in col_names:
                continue
            out[key] = self._sanitize_value(val, insp_cfg)
        return out

    def _sanitize_value(self, value: Any, cfg: InspectionConfig) -> Any:
        """将任意 DB 值转为 JSON 可序列化对象。"""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, (dict, list)):
            try:
                return json.loads(json.dumps(value, default=str, ensure_ascii=False))
            except (TypeError, ValueError):
                return str(value)
        text_repr = str(value)
        if cfg.truncate_large_fields and len(text_repr) > cfg.truncate_threshold:
            text_repr = text_repr[: cfg.truncate_threshold] + f"...<truncated, total {len(text_repr)} chars>"
        return text_repr


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------

class ConsoleReporter:
    """控制台表格输出器(委托 DatabaseSnapshot 渲染)。"""

    def __init__(self, config: DbInspectorConfig) -> None:
        """保存配置。"""
        self._config = config

    def print(self, snapshot: DatabaseSnapshot) -> None:
        """将快照以人类可读表格形式打印到 stdout。"""
        self._section(f"Database Snapshot: {snapshot.database_name}")
        print(f"  Generated at : {snapshot.generated_at}")
        print(f"  Schemas      : {', '.join(snapshot.schemas) if snapshot.schemas else '(none)'}")
        print(f"  Table count  : {len(snapshot.tables)}")

        for table in snapshot.tables:
            self._section(f"Table: {table.schema}.{table.name}"
                          + (f"  -- {table.comment}" if table.comment else ""))
            print(f"  Estimated rows: {table.row_count}")

            self._subsection("Columns")
            self._print_table(
                headers=["Name", "Type", "Nullable", "PK", "Default", "Comment"],
                rows=[
                    [
                        c.name, c.type, "YES" if c.nullable else "NO",
                        "✓" if c.is_primary_key else "",
                        c.default or "", c.comment or "",
                    ]
                    for c in table.columns
                ],
            )

            if self._config.inspection.include_indexes and table.indexes:
                self._subsection("Indexes")
                self._print_table(
                    headers=["Name", "Columns", "Unique", "Primary"],
                    rows=[
                        [idx.name, ", ".join(idx.columns),
                         "✓" if idx.unique else "", "✓" if idx.is_primary else ""]
                        for idx in table.indexes
                    ],
                )

            if self._config.inspection.include_constraints and table.constraints:
                self._subsection("Constraints")
                self._print_table(
                    headers=["Name", "Type", "Columns", "Ref Table", "Ref Columns"],
                    rows=[
                        [
                            c.name, c.type, ", ".join(c.columns),
                            c.referred_table or "",
                            ", ".join(c.referred_columns) if c.referred_columns else "",
                        ]
                        for c in table.constraints
                    ],
                )

            if self._config.inspection.include_table_data and table.rows:
                self._subsection(
                    f"Rows (showing {min(len(table.rows), self._config.output.max_rows_per_table_console)}"
                    f" of {table.row_count})"
                )
                display_rows = table.rows[: self._config.output.max_rows_per_table_console]
                col_names = [c.name for c in table.columns]
                self._print_table(
                    headers=col_names,
                    rows=[[str(r.get(col, "")) for col in col_names] for r in display_rows],
                )

    # ---- 排版辅助 ---------------------------------------------------------

    @staticmethod
    def _section(title: str) -> None:
        """打印醒目分隔标题。"""
        bar = "=" * max(60, len(title) + 4)
        print(f"\n{bar}")
        print(f"  {title}")
        print(bar)

    @staticmethod
    def _subsection(title: str) -> None:
        """打印子标题。"""
        print(f"\n-- {title} --")

    @staticmethod
    def _print_table(headers: list[str], rows: list[list[str]]) -> None:
        """简易表格打印,根据内容自动计算列宽。"""
        if not headers:
            return
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], min(len(str(cell)), 80))
        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header_row = "|" + "|".join(f" {h:<{widths[i]}} " for i, h in enumerate(headers)) + "|"
        print(sep)
        print(header_row)
        print(sep)
        for row in rows:
            cells = [str(row[i]) if i < len(row) else "" for i in range(len(headers))]
            cells = [c if len(c) <= widths[i] else c[: widths[i] - 3] + "..."
                     for i, c in enumerate(cells)]
            print("|" + "|".join(f" {c:<{widths[i]}} " for i, c in enumerate(cells)) + "|")
        if rows:
            print(sep)


class JsonReporter:
    """JSON 文件输出器。"""

    def __init__(self, config: DbInspectorConfig) -> None:
        """保存配置。"""
        self._config = config

    def write(self, snapshot: DatabaseSnapshot) -> Path:
        """将快照写入 JSON 文件,返回文件路径。"""
        out_dir = PROJECT_ROOT / self._config.output.json_output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"db_snapshot_{snapshot.database_name}_{ts}.json"
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        logger.info("JSON snapshot written to %s", path)
        return path
