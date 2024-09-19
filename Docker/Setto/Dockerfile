# Utilizar la imagen base de TensorFlow con GPU
FROM tensorflow/tensorflow:latest-gpu-py3

# Instalar Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Establecer el directorio de trabajo
WORKDIR /app

# Pasar argumentos a Docker
ARG CLOUD_NAME
ARG API_KEY
ARG API_SECRET

# Definir variables de entorno
ENV CLOUD_NAME=$CLOUD_NAME
ENV API_KEY=$API_KEY
ENV API_SECRET=$API_SECRET

# Instalar las dependencias del sistema
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libgtk-3-0 && \
    rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt al contenedor
COPY ./Docker/Setto/requirements.txt /app/

# Actualizar pip y las dependencias
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir jupyter "fastapi[standard]"

# Exponer el puerto para Jupyter
EXPOSE 8888

# Comando para iniciar Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
#docker run --gpus all -it --rm -p 8888:8888 -v C:\Users\48113164\Documents\GitHub\SignAI-ML\Docker\Setto:/app signai
# docker image ls
# docker image rm nombreimage