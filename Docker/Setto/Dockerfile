FROM tensorflow/tensorflow:latest-gpu-py3

ARG CLOUD_NAME
ARG API_KEY
ARG API_SECRET

ENV CLOUD_NAME=$CLOUD_NAME
ENV API_KEY=$API_KEY
ENV API_SECRET=$API_SECRET

WORKDIR /app

# Agrega la clave GPG faltante para el repositorio de NVIDIA
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx \
                        libglib2.0-0 \
                        libgtk-3-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY ./Docker/Setto/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyter "fastapi[standard]"

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
