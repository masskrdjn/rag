#!/bin/bash
# Quick Deployment Script for RAG System on Ubuntu
# Usage: sudo bash deploy.sh

set -e

# Répertoire d'installation
RAG_DIR="/home/rag"

echo "=== RAG System - Quick Deployment Script ==="
echo "Installation directory: $RAG_DIR"

# Update system
echo "Step 1: Updating system..."
apt update && apt upgrade -y

# Install dependencies
echo "Step 2: Installing Python and dependencies..."
apt install -y python3 python3-pip python3-venv git curl

# Install Ollama
echo "Step 3: Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
echo "Step 4: Downloading AI models..."
ollama pull mistral:7b
ollama pull nomic-embed-text

# Setup application directory
echo "Step 5: Setting up application..."
mkdir -p $RAG_DIR
mkdir -p $RAG_DIR/data
cd $RAG_DIR

# Install Python packages globally
echo "Step 6: Installing Python packages..."
pip3 install --upgrade pip
pip3 install langchain langchain-community langchain-chroma langchain-ollama chromadb fastapi uvicorn python-dotenv beautifulsoup4

echo "=== Deployment Complete ==="
echo "Next steps:"
echo "1. Copy your code files to $RAG_DIR/"
echo "2. Copy your HTML data to $RAG_DIR/data/"
echo "3. Run ingestion: cd $RAG_DIR && python3 ingest_html_adaptive.py"
echo "4. Start the service: bash $RAG_DIR/restart_server.sh"
