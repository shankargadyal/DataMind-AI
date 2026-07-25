# DataMind V.01 — production container image
FROM python:3.11-slim

# System deps: build tools needed for catboost/lightgbm/shap native wheels on some platforms
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Don't bake secrets or local dev artifacts into the image
RUN rm -f apikey.txt datamind.db *.log

RUN mkdir -p /app/uploads

# By default the app writes datamind.db inside the container, which is wiped
# on every redeploy. Mount a volume at /app (or set DATABASE_URL to a real
# Postgres instance) if you need user accounts/history to survive restarts —
# see ARCHITECTURE.md §4.
ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

EXPOSE 5000

# gunicorn is already in requirements.txt — use it instead of the Flask dev server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
