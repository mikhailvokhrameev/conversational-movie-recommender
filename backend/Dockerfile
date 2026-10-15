FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

# The venv lives outside /app so `COPY . .` and any bind mount can't shadow it.
# torch comes from the cu121 index (sm_61 support) -- see backend/pyproject.toml.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-cache

COPY . .

EXPOSE 8000

CMD ["uvicorn", "recommender.asgi:application", "--host", "0.0.0.0", "--port", "8000"]
