#!/bin/bash
#
# Réingestion ChromaDB avec chunking adaptatif (ingest_html_adaptive.py).
# Pensé pour Linux. Sous Windows, lancer directement :
#     python ingest_html_adaptive.py
#
# Variables d'environnement respectées :
#   RAG_CHROMA_DB_PATH : dossier ChromaDB cible (défaut /home/rag/chroma_db)
#   PROJECT_DIR        : dossier du projet (défaut /home/rag)

set -e

CHROMA_PATH="${RAG_CHROMA_DB_PATH:-/home/rag/chroma_db}"
PROJECT_DIR="${PROJECT_DIR:-/home/rag}"

echo "============================================================================"
echo "RÉINGESTION ADAPTATIVE — ChromaDB : $CHROMA_PATH"
echo "============================================================================"
echo ""
echo "Cette opération va REMPLACER la base de données actuelle."
echo "Une sauvegarde horodatée sera créée à côté."
echo ""
read -p "Continuer ? (oui/non) : " confirm
if [ "$confirm" != "oui" ]; then
    echo "Annulé."
    exit 0
fi

if [ -d "$CHROMA_PATH" ]; then
    BACKUP_DIR="${CHROMA_PATH}_backup_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "[1/3] Sauvegarde -> $BACKUP_DIR"
    cp -r "$CHROMA_PATH" "$BACKUP_DIR"

    echo "[2/3] Suppression de l'ancienne base"
    rm -rf "$CHROMA_PATH"
else
    echo "[1/3] Pas de base existante à sauvegarder"
    BACKUP_DIR=""
fi

echo "[3/3] Ingestion adaptative"
cd "$PROJECT_DIR"
if ! python3 ingest_html_adaptive.py; then
    echo ""
    echo "ÉCHEC de l'ingestion."
    if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
        echo "Restauration de la sauvegarde..."
        rm -rf "$CHROMA_PATH"
        cp -r "$BACKUP_DIR" "$CHROMA_PATH"
        echo "Sauvegarde restaurée."
    fi
    exit 1
fi

echo ""
echo "============================================================================"
echo "Réingestion terminée."
echo "  Base : $CHROMA_PATH"
[ -n "$BACKUP_DIR" ] && echo "  Backup : $BACKUP_DIR"
echo ""
echo "Pensez à redémarrer le serveur RAG."
echo "============================================================================"
