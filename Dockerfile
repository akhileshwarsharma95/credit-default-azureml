# Streamlit front end (Azure Container Apps / any container host)
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home appuser
WORKDIR /app
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
COPY src/creditrisk src/creditrisk
COPY app app
ENV MODEL_DIR=/app/app/model PYTHONPATH=/app/src
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
