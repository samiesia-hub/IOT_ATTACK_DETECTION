FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY demo-requirements.txt ./
RUN pip install --no-cache-dir -r demo-requirements.txt
COPY demo_server.py ./
COPY demo/ ./demo/
COPY models/ ./models/
COPY data/processed/test.parquet ./data/processed/test.parquet
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8501
CMD ["python", "demo_server.py", "--host", "0.0.0.0", "--port", "8501"]
