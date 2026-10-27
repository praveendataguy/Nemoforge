# The NemoForge web app (dashboard + build API).
# Deployable as a Nebius Serverless Endpoint; agents call Nemotron via Token Factory, so no GPU needed.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends util-linux && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nemoforge/ nemoforge/
COPY web/ web/
ENV NF_BUILDS_DIR=/data/builds
EXPOSE 8000
CMD ["uvicorn", "nemoforge.server:app", "--host", "0.0.0.0", "--port", "8000"]
