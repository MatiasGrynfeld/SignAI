FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ARG CLOUD_NAME
ARG API_KEY
ARG API_SECRET

ENV CLOUD_NAME=$CLOUD_NAME
ENV API_KEY=$API_KEY
ENV API_SECRET=$API_SECRET

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir jupyter "fastapi[standard]"
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libgtk-3-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

# docker-compose build
# docker run --gpus all -it --rm -p 8888:8888 -v D:\SignAI-ML\Docker\Mati:/app signai
# docker run --gpus all -it --rm -p 8888:8888 -v C:\Users\48519558\Desktop\SignAI-ML\Docker\Mati:/app signai

# docker image ls
# docker image rm nombreimage