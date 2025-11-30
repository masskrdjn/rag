#!/bin/bash
# =============================================================================
# diagnose_rag.sh - Diagnostic complet du système RAG
# =============================================================================

echo "🔍 Diagnostic du système RAG"
echo "============================"
echo ""

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 1. Vérification du serveur
echo -e "${BLUE}[1] État du serveur${NC}"
echo "-------------------"
if ss -tlnp 2>/dev/null | grep -q ":8000"; then
    PID=$(ss -tlnp 2>/dev/null | grep ":8000" | grep -oP 'pid=\K[0-9]+')
    echo -e "${GREEN}✓ Serveur actif sur le port 8000 (PID: $PID)${NC}"
    
    # Test health
    HEALTH=$(curl -s --max-time 5 http://localhost:8000/health 2>/dev/null)
    if echo "$HEALTH" | grep -q "healthy"; then
        echo -e "${GREEN}✓ Health check: OK${NC}"
    else
        echo -e "${YELLOW}⚠ Health check: Pas de réponse${NC}"
    fi
else
    echo -e "${RED}❌ Serveur NON actif sur le port 8000${NC}"
fi
echo ""

# 2. Processus Python
echo -e "${BLUE}[2] Processus Python/Uvicorn${NC}"
echo "----------------------------"
PROCS=$(ps aux | grep -E "python3.*(server|uvicorn)" | grep -v grep)
if [ -n "$PROCS" ]; then
    echo "$PROCS" | while read line; do
        PID=$(echo "$line" | awk '{print $2}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf $i" "}')
        echo -e "${GREEN}✓ PID $PID: $CMD${NC}"
    done
else
    echo -e "${YELLOW}⚠ Aucun processus serveur trouvé${NC}"
fi
echo ""

# 3. Base de données ChromaDB
echo -e "${BLUE}[3] Base de données ChromaDB${NC}"
echo "----------------------------"
if [ -d "/home/rag/chroma_db" ]; then
    SIZE=$(du -sh /home/rag/chroma_db 2>/dev/null | cut -f1)
    FILES=$(find /home/rag/chroma_db -type f 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Chemin: /home/rag/chroma_db${NC}"
    echo -e "${GREEN}✓ Taille: $SIZE${NC}"
    echo -e "${GREEN}✓ Fichiers: $FILES${NC}"
    
    # Vérifier sqlite
    if [ -f "/home/rag/chroma_db/chroma.sqlite3" ]; then
        SQLITE_SIZE=$(du -sh /home/rag/chroma_db/chroma.sqlite3 | cut -f1)
        echo -e "${GREEN}✓ SQLite: $SQLITE_SIZE${NC}"
    fi
else
    echo -e "${RED}❌ Base de données non trouvée!${NC}"
    echo "   Exécutez: python3 /home/rag/ingest_html_adaptive.py"
fi
echo ""

# 4. Fichiers de configuration
echo -e "${BLUE}[4] Fichiers du projet${NC}"
echo "----------------------"
FILES_TO_CHECK=(
    "/home/rag/server.py"
    "/home/rag/rag_pipeline.py"
    "/home/rag/ingest_html_adaptive.py"
)
for f in "${FILES_TO_CHECK[@]}"; do
    if [ -f "$f" ]; then
        MOD=$(stat -c %y "$f" 2>/dev/null | cut -d. -f1)
        echo -e "${GREEN}✓ $(basename $f) (modifié: $MOD)${NC}"
    else
        echo -e "${RED}❌ $f manquant${NC}"
    fi
done
echo ""

# 5. Dépendances Python
echo -e "${BLUE}[5] Dépendances Python${NC}"
echo "----------------------"
DEPS=("fastapi" "uvicorn" "langchain-chroma" "langchain-ollama" "chromadb")
for dep in "${DEPS[@]}"; do
    if pip3 show "$dep" &>/dev/null; then
        VERSION=$(pip3 show "$dep" 2>/dev/null | grep Version | cut -d: -f2 | tr -d ' ')
        echo -e "${GREEN}✓ $dep ($VERSION)${NC}"
    else
        echo -e "${RED}❌ $dep non installé${NC}"
    fi
done
echo ""

# 6. Ollama
echo -e "${BLUE}[6] Ollama${NC}"
echo "----------"
if command -v ollama &>/dev/null; then
    if ollama list 2>/dev/null | grep -q "llama3.2"; then
        echo -e "${GREEN}✓ llama3.2 disponible${NC}"
    else
        echo -e "${YELLOW}⚠ llama3.2 non trouvé${NC}"
    fi
    if ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
        echo -e "${GREEN}✓ nomic-embed-text disponible${NC}"
    else
        echo -e "${YELLOW}⚠ nomic-embed-text non trouvé${NC}"
    fi
else
    echo -e "${RED}❌ Ollama non installé${NC}"
fi
echo ""

# 7. Test de requête
echo -e "${BLUE}[7] Test de requête${NC}"
echo "-------------------"
if ss -tlnp 2>/dev/null | grep -q ":8000"; then
    RESPONSE=$(curl -s --max-time 30 -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "test"}' 2>/dev/null)
    
    if echo "$RESPONSE" | grep -q '"answer"'; then
        echo -e "${GREEN}✓ API répond correctement${NC}"
        # Vérifier le nouveau format
        if echo "$RESPONSE" | grep -q "Section [0-9]/[0-9]"; then
            echo -e "${GREEN}✓ Nouveau format de chunks (hiérarchique)${NC}"
        else
            echo -e "${YELLOW}⚠ Ancien format de chunks détecté${NC}"
            echo "   → Redémarrez le serveur: ./restart_rag.sh"
        fi
    else
        echo -e "${RED}❌ Réponse invalide${NC}"
        echo "$RESPONSE" | head -c 200
    fi
else
    echo -e "${YELLOW}⚠ Serveur non accessible, test ignoré${NC}"
fi
echo ""

# Résumé
echo "============================"
echo -e "${BLUE}📋 Résumé${NC}"
echo "============================"
if ss -tlnp 2>/dev/null | grep -q ":8000" && [ -d "/home/rag/chroma_db" ]; then
    echo -e "${GREEN}✅ Système opérationnel${NC}"
else
    echo -e "${RED}❌ Problèmes détectés - voir ci-dessus${NC}"
    echo ""
    echo "Actions recommandées:"
    if ! [ -d "/home/rag/chroma_db" ]; then
        echo "  1. python3 /home/rag/ingest_html_adaptive.py"
    fi
    if ! ss -tlnp 2>/dev/null | grep -q ":8000"; then
        echo "  2. ./restart_rag.sh"
    fi
fi
