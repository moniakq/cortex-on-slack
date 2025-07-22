FROM python:3.10-slim-buster
WORKDIR /app

COPY app_local.py ./
COPY cortex_chat.py ./
COPY generate_jwt.py ./
COPY requirements.txt ./
COPY rsa_key.p8 ./
COPY rsa_key.pub ./
COPY .env ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python3", "app_local.py"]