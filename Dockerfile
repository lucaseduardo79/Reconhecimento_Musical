# ─────────────────────────────────────────────────────────────────────────────
# Base: PyTorch 2.5 + CUDA 12.4 (compatível com drivers CUDA 12.x)
# GTX 1660 SUPER (Turing — compute capability 7.5) ✅
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL maintainer="lucaseduardo79"
LABEL description="OMR + Transposição de Partituras com IA"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV MUSIC21_MUSICXML_PATH=/usr/bin/musescore3

# ─── Dependências do sistema ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    musescore3 \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

# ─── Dependências Python ─────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Configura music21 para usar MuseScore ───────────────────────────────────
RUN python3 -c "from music21 import environment; us = environment.UserSettings(); us['musicxmlPath'] = '/usr/bin/musescore3'; us['musescoreDirectPNGPath'] = '/usr/bin/musescore3'"

# ─── Salva o caminho dos checkpoints para montar como volume ─────────────────
# Os modelos ONNX (~400 MB) são baixados na primeira execução via start.sh
ENV OEMER_CHECKPOINT_DIR=/opt/conda/lib/python3.11/site-packages/oemer/checkpoints

COPY partitura_ia.py .
COPY Tarantella_napoletana.jpeg .

COPY start.sh /start.sh
RUN sed -i 's/\r//' /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
