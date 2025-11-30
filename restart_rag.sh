#!/bin/bash
# =============================================================================
# restart_rag.sh - Redémarrage propre du serveur RAG
# =============================================================================

set -e

echo "🔄 Redémarrage du serveur RAG..."
echo "================================"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Arrêter les processus existants
echo -e "\n${YELLOW}[1/4] Arrêt des processus existants...${NC}"
pkill -9 -f "python3 server.py" 2>/dev/null || true
pkill -9 -f "uvicorn" 2>/dev/null || true
fuser -k 8000/tcp 2>/dev/null || true
sleep 2

# Vérifier que le port est libéré
if ss -tlnp | grep -q ":8000"; then
    echo -e "${RED}❌ Le port 8000 est toujours occupé${NC}"
    ss -tlnp | grep 8000
    exit 1
fi
echo -e "${GREEN}✓ Processus arrêtés${NC}"

# 2. Vérifier la base de données
echo -e "\n${YELLOW}[2/4] Vérification de la base ChromaDB...${NC}"
if [ -d "/home/rag/chroma_db" ]; then
    CHUNKS=$(find /home/rag/chroma_db -name "*.bin" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Base de données trouvée${NC}"
else
    echo -e "${RED}❌ Base de données non trouvée. Lancez d'abord:${NC}"
    echo "   python3 /home/rag/ingest_html_adaptive.py"
    exit 1
fi

# 3. Démarrer le serveur
echo -e "\n${YELLOW}[3/4] Démarrage du serveur...${NC}"
cd /home/rag
nohup python3 server.py > /tmp/rag_server.log 2>&1 &
SERVER_PID=$!
echo "PID: $SERVER_PID"

# 4. Attendre et vérifier
echo -e "\n${YELLOW}[4/4] Vérification du démarrage...${NC}"
for i in {1..15}; do
    sleep 1
    if ss -tlnp | grep -q ":8000"; then
        echo -e "\n${GREEN}✅ Serveur RAG démarré avec succès!${NC}"
        echo ""
        echo "📍 API disponible sur: http://localhost:8000"
        echo "📍 Health check:       http://localhost:8000/health"
        echo "📍 Logs:               tail -f /tmp/rag_server.log"
        echo ""
        
        # Test rapide
        if curl -s http://localhost:8000/health | grep -q "healthy"; then
            echo -e "${GREEN}✓ Health check OK${NC}"
        fi
        exit 0
    fi
    echo -n "."
done

echo -e "\n${RED}❌ Le serveur n'a pas démarré dans les temps${NC}"
echo "Vérifiez les logs: cat /tmp/rag_server.log"
cat /tmp/rag_server.log
exit 1
