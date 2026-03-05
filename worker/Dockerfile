FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached)
COPY worker/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Sling engine (shared with Vercel gateway)
COPY core/ ./core/
COPY config/ ./config/

# Copy worker source
COPY worker/redis_store.py ./redis_store.py
COPY worker/processor.py   ./processor.py
COPY worker/app.py         ./app.py

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
