FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib
COPY servir.py .
COPY modelos ./modelos
CMD ["sh", "-c", "uvicorn servir:app --host 0.0.0.0 --port ${PORT:-8000}"]
