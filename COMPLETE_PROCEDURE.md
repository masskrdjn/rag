# RAG System - Complete Installation & Operation Procedure

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Data Ingestion](#data-ingestion)
4. [Service Management](#service-management)
5. [Troubleshooting](#troubleshooting)
6. [Maintenance](#maintenance)

---

## Prerequisites

- **OS:** Ubuntu 22.04 or later (tested on WSL2)
- **Privileges:** sudo access
- **Network:** Internet connection for downloading models (~2-4 GB)
- **Disk Space:** ~10 GB free (for models and ChromaDB)

---

## Installation Steps

### Method 1: Automated Installation (Recommended)

```bash
# Download and execute the complete installation script
sudo bash complete_install.sh
```

This script will:
1. Install system dependencies (Python, pip, git, curl)
2. Install Ollama
3. Create `ragapp` user
4. Download AI models (llama3.2, nomic-embed-text)
5. Setup Python virtual environment
6. Ingest sample data
7. Start the service
8. Verify the installation

### Method 2: Manual Step-by-Step Installation

#### Step 1: System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and tools
sudo apt install -y python3 python3-pip python3-venv git curl

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

#### Step 2: User Creation
```bash
# Create dedicated user
sudo useradd -m -s /bin/bash ragapp

# Ensure proper ownership
sudo chown ragapp:ragapp /home/ragapp
```

#### Step 3: Download AI Models
```bash
# Pull LLM model (for generating answers)
sudo -u ragapp ollama pull llama3.2

# Pull embedding model (for vectorization)
sudo -u ragapp ollama pull nomic-embed-text
```

#### Step 4: Application Setup
```bash
# Create application directory
sudo -u ragapp mkdir -p /home/ragapp/rag-system
sudo -u ragapp mkdir -p /home/ragapp/rag-system/data

# Copy application files
sudo cp server.py rag_pipeline.py ingest.py /home/ragapp/rag-system/
sudo chown -R ragapp:ragapp /home/ragapp/rag-system
```

#### Step 5: Python Environment
```bash
cd /home/ragapp/rag-system

# Create virtual environment
sudo -u ragapp python3 -m venv venv

# Install dependencies
sudo -u ragapp venv/bin/pip install --upgrade pip
sudo -u ragapp venv/bin/pip install \
    langchain \
    langchain-community \
    langchain-chroma \
    langchain-ollama \
    chromadb \
    fastapi \
    uvicorn \
    python-dotenv
```

---

## Data Ingestion

### Initial Data Ingestion

1. **Place your documents** in `/home/ragapp/rag-system/data/`:
   ```bash
   # Example: Create sample data
   sudo -u ragapp bash -c "echo 'Your text content here' > /home/ragapp/rag-system/data/document.txt"
   ```

2. **Run the ingestion script**:
   ```bash
   sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest.py"
   ```

   **Expected output:**
   ```
   Loading documents from data...
   Loaded X documents.
   Split into Y chunks.
   Saving to ChromaDB...
   Ingestion complete! Data saved to chroma_db
   ```

### What Happens During Ingestion

The `ingest.py` script performs the following:

1. **Loading:** Reads all `.txt` files from the `data/` directory
2. **Chunking:** Splits documents into manageable pieces:
   - Chunk size: 1000 characters
   - Overlap: 200 characters (for context continuity)
3. **Embedding:** Converts text chunks to vectors using `nomic-embed-text` model
4. **Storage:** Saves vectors to ChromaDB in the `chroma_db/` directory

### Adding New Data

When you add new documents:

```bash
# 1. Add files to data directory
sudo cp /path/to/new_documents/*.txt /home/ragapp/rag-system/data/
sudo chown ragapp:ragapp /home/ragapp/rag-system/data/*.txt

# 2. Re-run ingestion
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest.py"

# 3. Restart the service
sudo pkill -f uvicorn
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

---

## Service Management

### Starting the Service

**Method A: Using nohup (Simple)**
```bash
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

**Method B: Using systemd (Production)**
```bash
# 1. Copy service file
sudo cp rag-api.service /etc/systemd/system/

# 2. Reload systemd
sudo systemctl daemon-reload

# 3. Enable and start
sudo systemctl enable rag-api
sudo systemctl start rag-api
```

### Stopping the Service

```bash
# Method A: Kill process
sudo pkill -f uvicorn

# Method B: Systemd
sudo systemctl stop rag-api
```

### Checking Service Status

```bash
# Check if process is running
ps aux | grep uvicorn

# Check logs (nohup method)
sudo cat /home/ragapp/rag-system/server.log

# Check logs (systemd method)
sudo journalctl -u rag-api -f
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Your question here"}'
```

---

## Troubleshooting

### Problem 1: Permission Denied Errors

**Symptom:**
```
bash: /home/ragapp/rag-system/file.txt: Permission denied
```

**Solution:**
```bash
# Fix ownership
sudo chown -R ragapp:ragapp /home/ragapp/rag-system

# Ensure parent directory ownership
sudo chown ragapp:ragapp /home/ragapp
```

### Problem 2: ChromaDB Error "Nothing found on disk"

**Symptom:**
```
Error creating hnsw segment reader: Nothing found on disk
```

**Cause:** Service started before data ingestion, or data ingestion failed.

**Solution:**
```bash
# 1. Stop the service
sudo pkill -f uvicorn

# 2. Verify data exists
ls -la /home/ragapp/rag-system/data/

# 3. Re-run ingestion
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest.py"

# 4. Restart service
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

### Problem 3: Generic Answers (Not Using RAG Data)

**Symptom:** API responds but doesn't use ingested data.

**Diagnosis:**
```bash
# Check if ChromaDB exists
ls -la /home/ragapp/rag-system/chroma_db/

# Check ingestion logs
sudo cat /home/ragapp/rag-system/server.log
```

**Solution:** Follow steps in Problem 2.

### Problem 4: Ollama Models Not Found

**Symptom:**
```
Model 'llama3.2' not found
```

**Solution:**
```bash
# List available models
sudo -u ragapp ollama list

# Pull missing models
sudo -u ragapp ollama pull llama3.2
sudo -u ragapp ollama pull nomic-embed-text
```

### Problem 5: Port 8000 Already in Use

**Symptom:**
```
ERROR: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or change port in startup command
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8001 > server.log 2>&1 &"
```

### Problem 6: Python Dependencies Missing

**Symptom:**
```
ModuleNotFoundError: No module named 'langchain'
```

**Solution:**
```bash
# Reinstall dependencies
sudo -u ragapp /home/ragapp/rag-system/venv/bin/pip install \
    langchain langchain-community langchain-chroma \
    langchain-ollama chromadb fastapi uvicorn python-dotenv
```

---

## Maintenance

### Monitoring

```bash
# Check server logs
sudo tail -f /home/ragapp/rag-system/server.log

# Monitor resource usage
htop
```

### Backup

```bash
# Backup data and database
sudo tar -czf rag-backup-$(date +%Y%m%d).tar.gz \
    /home/ragapp/rag-system/data \
    /home/ragapp/rag-system/chroma_db
```

### Updating Models

```bash
# Update Ollama models
sudo -u ragapp ollama pull llama3.2
sudo -u ragapp ollama pull nomic-embed-text

# Restart service to use updated models
sudo pkill -f uvicorn
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

### Clearing and Re-ingesting Data

```bash
# Stop service
sudo pkill -f uvicorn

# Remove old database
sudo rm -rf /home/ragapp/rag-system/chroma_db

# Clear data directory
sudo rm -rf /home/ragapp/rag-system/data/*

# Add new data
sudo cp /path/to/new/data/*.txt /home/ragapp/rag-system/data/
sudo chown ragapp:ragapp /home/ragapp/rag-system/data/*.txt

# Re-ingest
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest.py"

# Restart service
sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
```

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Start service | `sudo -u ragapp bash -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"` |
| Stop service | `sudo pkill -f uvicorn` |
| Check status | `ps aux \| grep uvicorn` |
| View logs | `sudo cat /home/ragapp/rag-system/server.log` |
| Test API | `curl http://localhost:8000/health` |
| Ingest data | `sudo -u ragapp bash -c "cd /home/ragapp/rag-system && venv/bin/python3 ingest.py"` |
| List models | `sudo -u ragapp ollama list` |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                    Client (curl/browser)                 │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST /ask
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Server (server.py)                  │
│                  Port 8000                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│           RAG Pipeline (rag_pipeline.py)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   ChromaDB   │  │  Embeddings  │  │   LLM Model  │  │
│  │  (Retriever) │  │ nomic-embed  │  │   llama3.2   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                         ▲
                         │ Ingestion (ingest.py)
                         │
┌─────────────────────────────────────────────────────────┐
│              Data Files (data/*.txt)                     │
└─────────────────────────────────────────────────────────┘
```

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs: `/home/ragapp/rag-system/server.log`
3. Verify all prerequisites are met
4. Ensure Ollama service is running: `systemctl status ollama`
