# Use official Python runtime as base image
FROM python:3.11-slim

# Set environment variables to prevent pyc files and buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Set directory in container
WORKDIR /app

# Install system dependencies (build-essential for compiling, libgomp1 for xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Expose port
EXPOSE 5000

# Command to run using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
