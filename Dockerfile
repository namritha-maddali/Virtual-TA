FROM python:3.12-slim

# for libvips to support pyvips
RUN apt-get update && \
    apt-get install -y libvips build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"