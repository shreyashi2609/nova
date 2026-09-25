# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Render injects its own PORT env var at runtime and routes traffic to it;
# 8080 is just the local fallback for testing the image directly.
EXPOSE 8080
ENV PORT=8080

# Shell form (not exec/list form) so $PORT is actually expanded at container
# start — Render sets this dynamically, and exec-form CMD would pass the
# literal string "$PORT" to uvicorn and fail to bind.
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}