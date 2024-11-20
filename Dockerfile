FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt



# RUN DOCKER
# docker-compose build
# docker-compose up -d