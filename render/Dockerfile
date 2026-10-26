# Live backend for Render's free tier (512 MB RAM): ONNX models, no PyTorch.
# Build context is the repository root (see render.yaml).
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/tmp/hf
WORKDIR /app
COPY render/requirements.txt .
RUN pip install -r requirements.txt
COPY src/pjassist ./src/pjassist
RUN python -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; from pjassist.onnx_models import fetch; fetch(Path('/app/models'))" && rm -rf /tmp/hf
COPY data/corpus/passages.jsonl ./data/corpus/passages.jsonl
COPY data/cache/passages_emb.npy ./data/cache/passages_emb.npy
COPY results/dev/selection.json ./results/dev/selection.json
COPY space/app.py ./app.py
ENV PJ_BACKEND=onnx PJ_MODELS=/app/models
EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
