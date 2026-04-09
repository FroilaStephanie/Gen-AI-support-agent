FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create data directories
RUN mkdir -p data/pdfs chroma_db

# Seed the database (idempotent — uses CREATE TABLE IF NOT EXISTS)
RUN python setup_db.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none", \
     "--server.headless=true"]
