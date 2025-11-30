#!/bin/bash
# Script pour redémarrer le serveur RAG

RAG_DIR="/home/rag"

echo "Redémarrage du serveur RAG..."

# Arrêter le serveur existant s'il y en a un
pkill -f 'python3 server.py' || true
sleep 2

# Démarrer le serveur
cd "$RAG_DIR" && nohup python3 server.py > server.log 2>&1 &

echo "Attente de 5 secondes..."
sleep 5

# Vérifier que le serveur est démarré
if ps aux | grep -q '[p]ython3 server.py'; then
    echo "✓ Serveur démarré"
    # Tester l'API
    sleep 2
    curl -s http://localhost:8000/health
    echo ""
else
    echo "✗ Erreur: serveur non démarré"
    echo "Consultez les logs: tail -50 $RAG_DIR/server.log"
    exit 1
fi
