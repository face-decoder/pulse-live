FROM python:3.12-slim

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/
RUN pip install --no-cache-dir uv && uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
