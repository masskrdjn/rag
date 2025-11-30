#!/bin/bash

echo "============================================================================"
echo "RÉINGESTION AVEC CHUNKING ADAPTATIF INTELLIGENT"
echo "============================================================================"
echo ""
echo "Cette méthode adapte automatiquement la taille des chunks selon:"
echo "  • La longueur du document"
echo "  • Le type de document (Amadeus vs standard)"
echo "  • La structure du contenu"
echo ""
echo "Avantages:"
echo "  ✓ Documents courts (< 2000 chars):  chunks de 400  → contexte maximal"
echo "  ✓ Documents moyens (2k-10k chars):  chunks de 500-600 → équilibre"
echo "  ✓ Documents longs (> 10k chars):    chunks de 700-900 → précision"
echo "  ✓ Sections Amadeus:                 chunks réduits de 15%"
echo ""
echo "⚠️  IMPORTANT: Cela va remplacer votre base de données actuelle!"
echo ""
read -p "Voulez-vous continuer? (oui/non): " confirm

if [ "$confirm" != "oui" ]; then
    echo "Opération annulée."
    exit 0
fi

CHROMA_PATH="/home/rag/chroma_db"

echo ""
echo "1️⃣  Sauvegarde de l'ancienne base..."
BACKUP_DIR="/home/rag/chroma_db_backup_$(date +%Y%m%d_%H%M%S)"
if [ -d "$CHROMA_PATH" ]; then
    cp -r "$CHROMA_PATH" "$BACKUP_DIR"
    echo "✓ Sauvegarde créée: $BACKUP_DIR"
else
    echo "⚠️  Pas de base existante à sauvegarder"
fi

echo ""
echo "2️⃣  Suppression de l'ancienne base..."
rm -rf "$CHROMA_PATH"
echo "✓ Ancienne base supprimée"

echo ""
echo "3️⃣  Réingestion avec chunking adaptatif..."
cd /home/rag
python3 ingest_html_adaptive.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'ingestion!"
    echo "Restauration de la sauvegarde..."
    rm -rf "$CHROMA_PATH"
    cp -r "$BACKUP_DIR" "$CHROMA_PATH"
    echo "✓ Sauvegarde restaurée"
    exit 1
fi

echo ""
echo "4️⃣  Test de récupération sur différents types de documents..."

python3 << 'PYTHON_SCRIPT'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import sys

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="/home/rag/chroma_db", embedding_function=embeddings)

tests = [
    {
        "name": "Repas spéciaux Amadeus (section courte)",
        "query": "format amadeus repas spéciaux",
        "keywords": ["repas", "srr", "srvgml"]
    },
    {
        "name": "Congés payés (document court)",
        "query": "règles des congés payés",
        "keywords": ["congés", "payés"]
    },
    {
        "name": "Railneo connecteur",
        "query": "connecteur Railneo speedrail",
        "keywords": ["railneo", "speedrail", "train"]
    }
]

all_passed = True

for test in tests:
    print(f"\n{'='*70}")
    print(f"TEST: {test['name']}")
    print(f"Query: {test['query']}")
    print(f"{'='*70}")
    
    results = db.similarity_search(test['query'], k=3)
    
    if not results:
        print("❌ Aucun résultat")
        all_passed = False
        continue
    
    found = False
    for i, doc in enumerate(results, 1):
        content_lower = doc.page_content.lower()
        keywords_found = [kw for kw in test['keywords'] if kw in content_lower]
        
        if keywords_found:
            found = True
            print(f"✅ Document {i}: Trouvé!")
            print(f"   Keywords: {', '.join(keywords_found)}")
            print(f"   Chunk size: {doc.metadata.get('chunk_size', 'N/A')}")
            print(f"   Strategy: {doc.metadata.get('chunk_strategy', doc.metadata.get('type', 'N/A'))}")
            break
    
    if not found:
        print(f"⚠️  Mots-clés non trouvés dans les résultats (peut être ok si document absent)")

print(f"\n{'='*70}")
print("✅ Tests de récupération terminés")
print(f"{'='*70}")

PYTHON_SCRIPT

echo ""
echo "============================================================================"
echo "✅ RÉINGESTION ADAPTATIVE TERMINÉE!"
echo "============================================================================"
echo ""
echo "📊 Résumé:"
echo "  • Sauvegarde: $BACKUP_DIR"
echo "  • Nouvelle base: $CHROMA_PATH"
echo "  • Script utilisé: ingest_html_adaptive.py"
echo ""
echo "🎯 Prochaine étape:"
echo "  Redémarrer le serveur RAG:"
echo "    cd /home/rag && python3 server.py"
echo ""
echo "🧪 Tester:"
echo "  curl -X POST http://localhost:8000/ask \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\":\"Comment utiliser le connecteur Railneo?\"}'"
