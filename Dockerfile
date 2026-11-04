# Serving image. The PyPI torch wheel bundles CUDA, so run with `--gpus all` for GPU
# inference; without a GPU it falls back to CPU.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra ml --extra serve
COPY src ./src
RUN uv sync --frozen --no-dev --extra ml --extra serve

ENV CHASHMA_MODEL=/models/chashma
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "chashma.serve.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
