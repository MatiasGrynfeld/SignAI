FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir jupyter "fastapi[standard]"

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libgtk-3-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "api.init:app", "--host", "0.0.0.0", "--port", "8000"]
