# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to compile some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Create required directories
RUN mkdir -p data models

# ── Generate data + train model at build time (bakes artefacts into image) ──
# Comment this out if you prefer to mount pre-trained artefacts via a volume.
RUN python run_pipeline.py

# Expose both API and Streamlit ports
EXPOSE 8000 8501

# Default: run FastAPI. Override CMD to run Streamlit.
# Example: docker run ... streamlit run app/streamlit_app.py --server.port 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
