FROM python:3.10

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r ./requirements.txt
RUN pip install --no-cache-dir "fastapi[standard]"

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libgtk-3-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

CMD ["uvicorn", "init:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# docker build .
# docker run -p 8000:8000 <image_id>