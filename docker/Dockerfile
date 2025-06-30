# File: Dockerfile
# Path: NeuroCluster-Elite/docker/Dockerfile
# Description: Multi-stage Docker container for NeuroCluster Elite Trading Platform
#
# This Dockerfile creates an optimized production container for the NeuroCluster Elite
# trading platform with the following features:
# - Multi-stage build for smaller final image
# - Python 3.11 slim base image for optimal performance
# - Non-root user for security
# - Health checks and monitoring
# - Optimized layer caching
# - Support for both development and production modes
#
# Build commands:
#   docker build -t neurocluster-elite:latest -f docker/Dockerfile .
#   docker build -t neurocluster-elite:dev --target development -f docker/Dockerfile .
#
# Run commands:
#   docker run -p 8501:8501 -p 8000:8000 neurocluster-elite:latest
#   docker run -p 8501:8501 -p 8000:8000 -v $(pwd)/data:/app/data neurocluster-elite:latest
#
# Author: Your Name
# Created: 2025-06-30
# Version: 1.0.0
# License: MIT

# ==================== BUILD STAGE ====================
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

# Add metadata labels
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="NeuroCluster Elite" \
      org.label-schema.description="Advanced algorithmic trading platform with 99.59% efficiency" \
      org.label-schema.url="https://github.com/neurocluster-elite/neurocluster-elite" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/neurocluster-elite/neurocluster-elite" \
      org.label-schema.vendor="NeuroCluster Elite" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0" \
      maintainer="neurocluster-elite@example.com"

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build dependencies
    build-essential \
    gcc \
    g++ \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # System utilities
    curl \
    wget \
    git \
    # Audio dependencies (for voice commands)
    libasound2-dev \
    portaudio19-dev \
    # Database dependencies
    sqlite3 \
    # Network utilities
    netcat-traditional \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Create app directory
WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY requirements.txt setup.py ./
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==================== DEVELOPMENT STAGE ====================
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    jupyter \
    ipython

# Copy source code
COPY . .

# Create directories
RUN python create_directories.py

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG \
    PAPER_TRADING=true

# Expose ports
EXPOSE 8501 8000 8888

# Development command
CMD ["streamlit", "run", "main_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# ==================== PRODUCTION STAGE ====================
FROM python:3.11-slim as production

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    PAPER_TRADING=true \
    TZ=UTC

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries
    libblas3 \
    liblapack3 \
    libatlas3-base \
    # Audio runtime (for voice commands)
    libasound2 \
    portaudio19-dev \
    # Database
    sqlite3 \
    # Network utilities for health checks
    curl \
    netcat-traditional \
    # Process management
    supervisor \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y

# Create non-root user for security
RUN groupadd -r neurocluster && useradd -r -g neurocluster neurocluster

# Create app directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=neurocluster:neurocluster . .

# Create required directories with proper permissions
RUN python create_directories.py && \
    chown -R neurocluster:neurocluster /app && \
    chmod +x /app/scripts/*.py 2>/dev/null || true && \
    chmod +x /app/main_*.py

# Create data directories with proper permissions
RUN mkdir -p /app/data/{cache,logs,exports,backups,databases} && \
    mkdir -p /app/logs/{application,trading,analysis,errors,audit} && \
    chown -R neurocluster:neurocluster /app/data /app/logs

# Create supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/neurocluster.conf

# Switch to non-root user
USER neurocluster

# Expose ports
EXPOSE 8501 8000

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Production command with supervisor
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/neurocluster.conf"]

# ==================== NVIDIA CUDA STAGE (Optional) ====================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as cuda-production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    ENVIRONMENT=production \
    CUDA_ENABLED=true

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    # Build tools
    build-essential \
    gcc \
    g++ \
    # System utilities
    curl \
    wget \
    git \
    # Audio dependencies
    libasound2-dev \
    portaudio19-dev \
    # Database
    sqlite3 \
    # Network utilities
    netcat \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-specific packages
RUN pip install --no-cache-dir \
    cupy-cuda11x \
    cudf-cu11 \
    cuml-cu11

# Copy application
COPY . .

# Create directories
RUN python create_directories.py

# Create non-root user
RUN groupadd -r neurocluster && useradd -r -g neurocluster neurocluster && \
    chown -R neurocluster:neurocluster /app

# Switch to non-root user
USER neurocluster

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# CUDA command
CMD ["python", "main_server.py", "--host", "0.0.0.0", "--port", "8000"]

# ==================== BUILD HOOKS ====================

# Build hook for automated builds
ONBUILD COPY . /app/
ONBUILD RUN python create_directories.py
ONBUILD RUN chown -R neurocluster:neurocluster /app

# ==================== USAGE EXAMPLES ====================

# Example build commands:
# 
# Build production image:
# docker build -t neurocluster-elite:latest --target production .
# 
# Build development image:
# docker build -t neurocluster-elite:dev --target development .
# 
# Build CUDA-enabled image:
# docker build -t neurocluster-elite:cuda --target cuda-production .
# 
# Build with build args:
# docker build \
#   --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
#   --build-arg VCS_REF=$(git rev-parse --short HEAD) \
#   --build-arg VERSION=1.0.0 \
#   -t neurocluster-elite:latest .
#
# Example run commands:
#
# Run production container:
# docker run -d \
#   --name neurocluster-elite \
#   -p 8501:8501 \
#   -p 8000:8000 \
#   -v $(pwd)/data:/app/data \
#   -v $(pwd)/logs:/app/logs \
#   --restart unless-stopped \
#   neurocluster-elite:latest
#
# Run with environment variables:
# docker run -d \
#   --name neurocluster-elite \
#   -p 8501:8501 \
#   -p 8000:8000 \
#   -e PAPER_TRADING=true \
#   -e LOG_LEVEL=INFO \
#   -e INITIAL_CAPITAL=100000 \
#   -v $(pwd)/data:/app/data \
#   neurocluster-elite:latest
#
# Run development container:
# docker run -it \
#   --name neurocluster-dev \
#   -p 8501:8501 \
#   -p 8000:8000 \
#   -p 8888:8888 \
#   -v $(pwd):/app \
#   neurocluster-elite:dev bash
#
# Run CUDA container:
# docker run -d \
#   --name neurocluster-cuda \
#   --gpus all \
#   -p 8501:8501 \
#   -p 8000:8000 \
#   -v $(pwd)/data:/app/data \
#   neurocluster-elite:cuda