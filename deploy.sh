#!/bin/bash

# Update dari GitHub
git pull origin main

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --no-cache-dir --default-timeout=1000

# Jalankan Docker dengan sudo
sudo docker-compose down
sudo docker-compose up --build -d
