#!/bin/bash

echo "============================================================================"
echo "RÉINGESTION AVEC CHUNKING ADAPTATIF INTELLIGENT"
echo "============================================================================"
echo ""
echo "Cette nouvelle méthode adapte automatiquement la taille des chunks selon:"
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

echo ""
echo "1️⃣  Sauvegarde de l'ancienne base..."
BACKUP_DIR="/home/ragapp/rag-system/chroma_db_backup_adaptive_$(date +%Y%m%d_%H%M%S)"
if [ -d "/home/ragapp/rag-system/chroma_db" ]; then
    cp -r /home/ragapp/rag-system/chroma_db "$BACKUP_DIR"
    echo "✓ Sauvegarde créée: $BACKUP_DIR"
else
    echo "⚠️  Pas de base existante à sauvegarder"
fi

echo ""
echo "2️⃣  Suppression de l'ancienne base..."
rm -rf /home/ragapp/rag-system/chroma_db
echo "✓ Ancienne base supprimée"

echo ""
echo "3️⃣  Réingestion avec chunking adaptatif..."
cd /home/rag
/home/ragapp/rag-system/venv/bin/python3 ingest_html_adaptive.py

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'ingestion!"
    echo "Restauration de la sauvegarde..."
    rm -rf /home/ragapp/rag-system/chroma_db
    cp -r "$BACKUP_DIR" /home/ragapp/rag-system/chroma_db
    echo "✓ Sauvegarde restaurée"
    exit 1
fi

echo ""
echo "4️⃣  Copie de la nouvelle base vers le répertoire de production..."
if [ -d "chroma_db" ]; then
    cp -r chroma_db /home/ragapp/rag-system/
    echo "✓ Base copiée vers /home/ragapp/rag-system/chroma_db"
fi

echo ""
echo "5️⃣  Test de récupération sur différents types de documents..."

/home/ragapp/rag-system/venv/bin/python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '/home/ragapp/rag-system/venv/lib/python3.10/site-packages')

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory="/home/ragapp/rag-system/chroma_db", embedding_function=embeddings)

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
        "name": "Documents longs (groupes)",
        "query": "procédure groupes fournisseurs",
        "keywords": ["groupe", "cotation"]
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
        print("❌ Mots-clés non trouvés dans les résultats")
        all_passed = False

print(f"\n{'='*70}")
if all_passed:
    print("✅ TOUS LES TESTS SONT PASSÉS!")
    print("Le chunking adaptatif fonctionne correctement.")
else:
    print("⚠️  Certains tests ont échoué")
    sys.exit(1)
print(f"{'='*70}")

PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "6️⃣  Redémarrage du serveur RAG..."
    sudo systemctl restart rag-api
    sleep 2
    sudo systemctl status rag-api --no-pager
    
    echo ""
    echo "============================================================================"
    echo "✅ RÉINGESTION ADAPTATIVE TERMINÉE AVEC SUCCÈS!"
    echo "============================================================================"
    echo ""
    echo "📊 Résumé:"
    echo "  • Sauvegarde: $BACKUP_DIR"
    echo "  • Nouvelle base: /home/ragapp/rag-system/chroma_db"
    echo "  • Chunking: Adaptatif selon taille de document"
    echo "  • Serveur: Redémarré"
    echo ""
    echo "🎯 Différences avec la version précédente:"
    echo "  • Documents très courts: chunk 400 au lieu de 600"
    echo "  • Documents longs: chunk 700-900 au lieu de 600"
    echo "  • Sections Amadeus: chunks réduits de 15%"
    echo "  • Meilleure précision sur tous types de documents"
    echo ""
    echo "🧪 Tester:"
    echo "  curl -X POST http://localhost:8000/ask \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"question\":\"Quel est le format amadeus pour les repas spéciaux ?\"}'"
else
    echo ""
    echo "⚠️  Les tests ont échoué. Voir les détails ci-dessus."
    exit 1
fi
