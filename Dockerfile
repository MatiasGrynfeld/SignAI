FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir jupyter "fastapi[standard]" uvicorn

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx \
                        libglib2.0-0 \
                        libgtk-3-0 && \
    rm -rf /var/lib/apt/lists/*

ARG CLOUD_NAME

ARG API_KEY

ARG API_SECRET

ENV CLOUD_NAME=$CLOUD_NAME

ENV API_KEY=$API_KEY

ENV API_SECRET=$API_SECRET

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]