FROM tensorflow/tensorflow:latest-gpu-py3

ARG CLOUD_NAME
ARG API_KEY
ARG API_SECRET

ENV CLOUD_NAME=$CLOUD_NAME
ENV API_KEY=$API_KEY
ENV API_SECRET=$API_SECRET

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy nltk opencv-python mediapipe fastapi cloudinary uvicorn python-dotenv "fastapi[standard]"

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx \
                        libglib2.0-0 \
                        libgtk-3-0 && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]


# docker-compose build
# docker run --gpus all -it --rm -p 8888:8888 -v C:\Users\48113164\Documents\GitHub\SignAI-ML\Docker\Setto:/app signai

# docker image ls
# docker image rm nombreimage