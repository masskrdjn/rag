#!/bin/bash
#
# Redémarrage du serveur RAG (FastAPI / uvicorn).
# Pensé pour Linux. Sous Windows, lancer directement :
#     python server.py
#
# Variables d'environnement respectées :
#   RAG_CHROMA_DB_PATH : dossier ChromaDB (défaut /home/rag/chroma_db)
#   PROJECT_DIR        : dossier du projet (défaut /home/rag)
#   RAG_PORT           : port d'écoute (défaut 8000)
#   RAG_LOG_FILE       : fichier de log (défaut $PROJECT_DIR/server.log)
#
# Pour un déploiement systemd, préférer :
#     sudo systemctl restart rag-api.service

set -e

CHROMA_PATH="${RAG_CHROMA_DB_PATH:-/home/rag/chroma_db}"
PROJECT_DIR="${PROJECT_DIR:-/home/rag}"
PORT="${RAG_PORT:-8000}"
LOG_FILE="${RAG_LOG_FILE:-$PROJECT_DIR/server.log}"

echo "============================================================================"
echo "REDÉMARRAGE RAG — projet : $PROJECT_DIR  |  port : $PORT"
echo "============================================================================"

# 1. Arrêt des processus existants
echo ""
echo "[1/4] Arrêt des processus existants"
pkill -f "python3 server.py" 2>/dev/null || true
pkill -f "uvicorn server:app" 2>/dev/null || true
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
sleep 2

if command -v ss >/dev/null 2>&1 && ss -tln | grep -q ":${PORT} "; then
    echo "ERREUR : le port ${PORT} est toujours occupé."
    ss -tlnp 2>/dev/null | grep ":${PORT} " || true
    exit 1
fi

# 2. Vérification de la base ChromaDB
echo "[2/4] Vérification de la base ChromaDB ($CHROMA_PATH)"
if [ ! -d "$CHROMA_PATH" ]; then
    echo "ERREUR : base introuvable. Lancer d'abord :"
    echo "    cd \"$PROJECT_DIR\" && python3 ingest_html_adaptive.py"
    exit 1
fi

# 3. Démarrage du serveur
echo "[3/4] Démarrage du serveur (logs : $LOG_FILE)"
cd "$PROJECT_DIR"
nohup python3 server.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "    PID : $SERVER_PID"

# 4. Healthcheck
echo "[4/4] Attente du /health"
for i in $(seq 1 15); do
    sleep 1
    if curl -fs "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo ""
        echo "============================================================================"
        echo "Serveur démarré."
        echo "  API    : http://localhost:${PORT}"
        echo "  Health : http://localhost:${PORT}/health"
        echo "  Logs   : tail -f $LOG_FILE"
        echo "============================================================================"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "ERREUR : le processus serveur s'est arrêté. Dernières lignes du log :"
        tail -n 30 "$LOG_FILE" 2>/dev/null || true
        exit 1
    fi
done

echo ""
echo "ERREUR : /health ne répond pas après 15s. Dernières lignes du log :"
tail -n 30 "$LOG_FILE" 2>/dev/null || true
exit 1
