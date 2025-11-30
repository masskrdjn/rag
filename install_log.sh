#!/bin/bash
# Journal d'installation
# Ce fichier contient toutes les commandes exécutées pendant le processus d'installation.
# Répertoire d'installation : /home/rag/

# 1. Exécuter le script de déploiement automatisé
sudo bash deploy.sh

# 2. Les fichiers de l'application sont déjà dans /home/rag/
# Fichiers principaux : server.py, rag_pipeline.py, ingest_html_adaptive.py

# 3. Configuration du service Systemd (optionnel)
# Cela configure l'API RAG pour qu'elle démarre automatiquement
sudo cp /home/rag/rag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rag-api
# sudo systemctl start rag-api

# 4. Installer les dépendances Python
pip3 install --upgrade pip
pip3 install langchain langchain-community langchain-chroma langchain-ollama chromadb fastapi uvicorn python-dotenv beautifulsoup4

# 5. Démarrer le service (méthode simple)
bash /home/rag/restart_server.sh

# 6. Ingestion des données
# Placez vos fichiers HTML dans /home/rag/data/
cd /home/rag && python3 ingest_html_adaptive.py

# 7. Tester l'API
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "test"}'







