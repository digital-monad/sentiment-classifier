FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /home

COPY src/ ./src
COPY uv.lock pyproject.toml .python-version ./
COPY sentiment_model ./sentiment_model

ENV UV_COMPILE_BYTECODE=1
ENV UV_SYSTEM_PYTHON=1

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
