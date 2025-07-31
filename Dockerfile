FROM python:3.10-slim-buster
WORKDIR /app

COPY app_docker_connman.py ./
COPY cortex_chat_docker.py ./
COPY generate_jwt.py ./
COPY requirements.txt ./
COPY rsa_key.p8 ./
COPY rsa_key.pub ./
COPY connection_manager.py ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python3", "app_docker_connman.py"]