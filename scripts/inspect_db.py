# 数据库数据探查脚本入口 - 访问 Postgres Docker 容器中的所有表结构、字段及记录
# 运行指南:
#   前置条件:
#     1. 启动 Postgres 容器: docker compose up -d postgres
#     2. 安装依赖: pip install -e .
#   运行方式:
#     python -m scripts.inspect_db
#     输出:
#       - 控制台表格 (stdout)
#       - JSON 文件: reports/db_inspect/db_snapshot_<dbname>_<timestamp>.json
#   配置:
#     修改 configs/db_inspector.yaml 调整输出格式、表过滤、行数限制等

import logging
import sys

from app.tools.db_inspector import (
    ConsoleReporter,
    DatabaseInspector,
    DatabaseSnapshot,
    JsonReporter,
    load_inspector_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """加载配置,执行探查,按配置格式输出结果。

    Returns:
        0 表示成功, 1 表示发生异常
    """
    config = load_inspector_config()
    logger.info("Loaded db_inspector config (output format=%s)", config.output.format)

    try:
        with DatabaseInspector(config) as insp:
            snapshot: DatabaseSnapshot = insp.inspect()
    except Exception as e:
        logger.error("Database inspection failed: %s", e)
        logger.exception("Full traceback:")
        print("\n[ERROR] 无法连接到 Postgres 容器,请确认:", file=sys.stderr)
        print("  1. 容器已启动: docker compose up -d postgres", file=sys.stderr)
        print("  2. 端口 5432 未被占用", file=sys.stderr)
        print("  3. .env 中 POSTGRES_URL 正确 (默认已指向 localhost:5432)", file=sys.stderr)
        return 1

    if config.output.format in {"table", "both"}:
        ConsoleReporter(config).print(snapshot)

    if config.output.format in {"json", "both"}:
        try:
            path = JsonReporter(config).write(snapshot)
            print(f"\n[OK] JSON 快照已保存到: {path}")
        except OSError as e:
            logger.error("Failed to write JSON snapshot: %s", e)
            return 1

    print("\n[OK] 数据库探查完成。")
    print(f"  - 数据库: {snapshot.database_name}")
    print(f"  - Schema 数量: {len(snapshot.schemas)}")
    print(f"  - 表数量: {len(snapshot.tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
