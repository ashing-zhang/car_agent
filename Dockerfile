# AutoAgent 生产镜像
# 构建指南: docker build -t autoagent:latest .
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 生产环境使用 COPY 将代码打包进镜像
COPY pyproject.toml ./
COPY app/ ./app/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000

CMD ["python", "-m", "app.main"]
