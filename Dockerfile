# Backend Dockerfile for Genesis Evolution Platform
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Copy application code
COPY shinka/ ./shinka/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Create directory for results and database
RUN mkdir -p /app/results /app/data

# Expose port for any potential API/web service
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GENESIS_DATA_DIR=/app/data
ENV GENESIS_RESULTS_DIR=/app/results

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "shinka.launch_hydra"]
