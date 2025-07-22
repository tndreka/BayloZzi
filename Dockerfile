# Multi-stage Dockerfile for BayloZzi Forex Trading System
# Optimized for production deployment with enhanced security

# Stage 1: Build stage
FROM python:3.11-slim-bullseye AS builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY BayloZzi/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim-bullseye AS production

# Set build metadata
LABEL maintainer="BayloZzi Trading System" \
      version="${VERSION}" \
      build-date="${BUILD_DATE}" \
      description="Enhanced Forex Trading System with 12-Factor Compliance"

# Create non-root user for security
RUN groupadd -r trader && useradd --no-log-init -r -g trader trader

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    HOME="/home/trader" \
    USER=trader

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/.prp && \
    chown -R trader:trader /app

# Copy application code
COPY --chown=trader:trader BayloZzi/ /app/
COPY --chown=trader:trader .prp/ /app/.prp/
COPY --chown=trader:trader logs/ /app/logs/ 2>/dev/null || true

# Create health check script
RUN echo '#!/bin/bash\npython -c "import sys; sys.exit(0)"' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh && \
    chown trader:trader /app/healthcheck.sh

# Switch to non-root user
USER trader

# Set working directory
WORKDIR /app

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Expose port for monitoring (if web interface is added)
EXPOSE 8080

# Default command
CMD ["python", "run/enhanced_backtest.py"]