FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# One COPY per directory: a multi-source COPY flattens directory
# contents under Buildah (verified on the RAG project, 18 Nov 2026).
COPY requirements.txt .
COPY app/ ./app/
COPY data/ ./data/
COPY artifacts/ ./artifacts/
COPY benchmarks/ ./benchmarks/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
