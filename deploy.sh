#!/bin/bash

# Navigate to project directory
cd /home/bot/chatbot-pmb-undiksha

# Pull the latest code from GitHub (redundant here since code is already copied)
git pull origin main

# Build the Docker image
docker build -t chatbot-pmb-undiksha .

# Stop and remove the old container if it exists
docker stop chatbot-pmb-undiksha || true
docker rm chatbot-pmb-undiksha || true

# Run the new container in detached mode
docker run -d --name chatbot-pmb-undiksha -p 1014:1014 chatbot-pmb-undiksha