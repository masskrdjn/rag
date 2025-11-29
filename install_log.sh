#!/bin/bash
# Installation Log
# This file contains all commands executed during the installation process.

# 1. Run automated deployment script
# Note: This script installs dependencies, creates user 'ragapp', and sets up the environment.
sudo bash deploy.sh

# 2. Copy application files
# We copy the source code to the application directory
sudo -u ragapp mkdir -p /home/ragapp/rag-system
sudo cp /home/rag/* /home/ragapp/rag-system/
sudo chown -R ragapp:ragapp /home/ragapp/rag-system

# 3. Setup Systemd Service
# This configures the RAG API to start automatically
sudo cp /home/rag/rag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rag-api
# sudo systemctl start rag-api # Will be run after deployment finishes

# 4. Fix Permissions (Required because of manual file copy)
# The previous copy command made files owned by root, causing deploy.sh to fail.
# CRITICAL: Also fix /home/ragapp itself - it must belong to ragapp!
sudo chown ragapp:ragapp /home/ragapp
sudo chown -R ragapp:ragapp /home/ragapp/rag-system


# Also ensure chroma_db directory (if exists) has correct permissions
# Or simply remove it and let the application recreate it with correct ownership
if [ -d /home/ragapp/rag-system/chroma_db ]; then
    sudo rm -rf /home/ragapp/rag-system/chroma_db
fi

sudo -u ragapp bash

# 5. Complete Installation (Manual Steps)
# Since deploy.sh failed, we finish the setup manually:
cd /home/ragapp/rag-system
python3 -m venv venv
/home/ragapp/rag-system/venv/bin/pip install --upgrade pip
/home/ragapp/rag-system/venv/bin/pip install langchain langchain-community langchain-chroma langchain-ollama chromadb fastapi uvicorn python-dotenv
exit

# 6. Start Service
sudo systemctl start rag-api

# 7. Create Sample Data (For Testing)
sudo -u ragapp mkdir -p /home/ragapp/rag-system/data
sudo -u ragapp bash -c "echo 'La Tour Eiffel mesure 330 mètres de hauteur. Elle a été construite par Gustave Eiffel pour l Exposition Universelle de 1889 à Paris.' > /home/ragapp/rag-system/data/sample.txt"

# NOTE: Systemd service may fail to start due to PATH issues.
# Manual start works perfectly:
# cd /home/ragapp/rag-system
# /home/ragapp/rag-system/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

# To run in background (alternative to systemd):
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"






