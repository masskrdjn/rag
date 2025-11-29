#!/bin/bash
# Quick Deployment Script for RAG System on Ubuntu
# Usage: sudo bash deploy.sh

set -e

echo "=== RAG System - Quick Deployment Script ==="

# Update system
echo "Step 1: Updating system..."
apt update && apt upgrade -y

# Install dependencies
echo "Step 2: Installing Python and dependencies..."
apt install -y python3 python3-pip python3-venv git curl

# Install Ollama
echo "Step 3: Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Create user
echo "Step 4: Creating ragapp user..."
if ! id -u ragapp > /dev/null 2>&1; then
    useradd -m -s /bin/bash ragapp
fi

# Pull models
echo "Step 5: Downloading AI models..."
sudo -u ragapp ollama pull llama3.2
sudo -u ragapp ollama pull nomic-embed-text

# Setup application directory
echo "Step 6: Setting up application..."
sudo -u ragapp mkdir -p /home/ragapp/rag-system
cd /home/ragapp/rag-system

# Create virtual environment
sudo -u ragapp python3 -m venv venv

# Install Python packages
sudo -u ragapp /home/ragapp/rag-system/venv/bin/pip install --upgrade pip
sudo -u ragapp /home/ragapp/rag-system/venv/bin/pip install langchain langchain-community langchain-chroma langchain-ollama chromadb fastapi uvicorn python-dotenv

echo "=== Deployment Complete ==="
echo "Next steps:"
echo "1. Copy your code files to /home/ragapp/rag-system/"
echo "2. Create systemd service (see DEPLOYMENT.md)"
echo "3. Start the service: sudo systemctl start rag-api"
