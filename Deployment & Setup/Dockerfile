# AI Safety Models POC - Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved logs static/css static/js

# Generate sample data
RUN python -c "\
from sample_data_generator import SampleDataGenerator; \
import pandas as pd; \
generator = SampleDataGenerator(); \
abuse_data = generator.generate_abuse_detection_data(100); \
crisis_data = generator.generate_crisis_data(50); \
content_data = generator.generate_content_filtering_data(30); \
abuse_data.to_csv('data/raw/sample_abuse_data.csv', index=False); \
crisis_data.to_csv('data/raw/sample_crisis_data.csv', index=False); \
content_data.to_csv('data/raw/sample_content_data.csv', index=False); \
print('Sample data generated')\
"

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run the application
CMD ["python", "app.py"]
