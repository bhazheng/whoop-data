FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY code/train.py /app/code/
COPY app/main.py /app/app/
COPY data/whoop_fitness_dataset_100k.csv /app/data/

# Create output directory for model
RUN mkdir -p /app/output/model

# Train the model during build (optional - uncomment if you want to train during build)
# RUN python train.py --max_depth 10 --n_estimators 100

# If not training during build, copy pre-trained model instead:
# COPY model_final.bin .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/')" || exit 1

# Run the prediction service with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]
