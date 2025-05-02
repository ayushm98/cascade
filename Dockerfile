FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.1

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev dependencies in production)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

# Copy source code
COPY src/ ./src/
COPY ml/ ./ml/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8000 8501

# Default command (API server)
CMD ["uvicorn", "cascade.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
