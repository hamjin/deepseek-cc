# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DB_PATH=/data/deepseek_anthropic_proxy.sqlite3 \
    LOG_DIR=/data/log

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin --no-create-home app

COPY pyproject.toml README.md LICENSE ./
COPY app ./app

RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple/

RUN python -m pip install --no-cache-dir --root-user-action=ignore .

RUN mkdir -p /data/log \
    && chown -R app:app /app /data

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/openapi.json', timeout=2).read(1)"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
