
# Stage 1: Build React frontend
FROM node:20-alpine AS frontend

WORKDIR /frontend

COPY templates/package*.json ./

RUN if [ -f package-lock.json ]; then \
      npm ci; \
    else \
      npm install; \
    fi

COPY templates/ ./

RUN npm run build

# Stage 2: Flask backend
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -f apikey.txt datamind.db *.log

RUN mkdir -p /app/uploads

# Use the freshly built React frontend
COPY --from=frontend /frontend/dist /app/templates/dist

ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=5000

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
