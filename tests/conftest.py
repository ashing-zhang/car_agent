# pytest 配置 - 确保无 __init__.py 时测试可发现 app 包
# 运行指南: pytest 自动加载此文件
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
