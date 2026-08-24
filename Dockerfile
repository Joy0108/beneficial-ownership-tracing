FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app/src

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY data ./data
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./

# Build the registers at image build time so the container starts screenable.
RUN python scripts/build_registers.py

ENTRYPOINT ["python", "-m", "ubo.cli"]
CMD ["eval"]
