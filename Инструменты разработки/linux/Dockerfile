# Проверка Herald на Linux (процессор, без окна) - Herald 3.2, по образцу Scribe 2.4.
#   docker build -t herald-linux -f "Инструменты разработки/linux/Dockerfile" .
#   docker run --rm -v "$PWD":/app -v herald-hf:/root/.cache herald-linux \
#       python3 "Инструменты разработки/smoke_pc.py"
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-tk ffmpeg libportaudio2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-pc.txt /tmp/requirements-pc.txt
RUN pip install --no-cache-dir -r /tmp/requirements-pc.txt
WORKDIR /app
