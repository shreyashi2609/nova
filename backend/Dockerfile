# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run FastAPI with Uvicorn (WebSockets supported)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]