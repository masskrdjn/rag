#!/bin/bash
# Script pour redémarrer le serveur RAG

echo "Redémarrage du serveur RAG..."

# Arrêter le serveur existant s'il y en a un
pkill -f uvicorn || true
sleep 2

# Démarrer le serveur
sudo su - ragapp -c "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"

echo "Attente de 5 secondes..."
sleep 5

# Vérifier que le serveur est démarré
if ps aux | grep -q '[u]vicorn server:app'; then
    echo "✓ Serveur démarré"
    # Tester l'API
    sleep 2
    curl -s http://localhost:8000/health
    echo ""
else
    echo "✗ Erreur: serveur non démarré"
    exit 1
fi
