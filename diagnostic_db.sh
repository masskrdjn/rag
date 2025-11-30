#!/bin/bash
echo "=========================================="
echo "DIAGNOSTIC BASE CHROMADB"
echo "=========================================="

cd /home/rag

echo ""
echo "1. Vérification de la base ChromaDB..."
if [ -f "chroma_db/chroma.sqlite3" ]; then
    size=$(du -h chroma_db/chroma.sqlite3 | cut -f1)
    echo "✓ Base présente (taille: $size)"
else
    echo "✗ Base introuvable!"
    exit 1
fi

echo ""
echo "2. Test de recherche dans la base..."
venv/bin/python3 << 'PYTHON_SCRIPT'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

count = vectorstore._collection.count()
print(f"✓ Total: {count} chunks")

if count == 0:
    print("✗ LA BASE EST VIDE!")
    exit(1)

print("\n3. Recherche: 'congés payés'")
results = vectorstore.similarity_search_with_score("congés payés", k=3)
print(f"✓ {len(results)} résultats trouvés\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"--- Résultat {i} (Distance: {score:.4f}) ---")
    source = doc.metadata.get('source', 'N/A')
    if 'data/' in source:
        source = source.split('data/')[-1]
    print(f"Source: {source}")
    content = doc.page_content.replace('\n', ' ')[:200]
    print(f"Contenu: {content}...")
    print()

print("\n4. Recherche du fichier '1068_Conges'")
all_docs = vectorstore.get()
conges_count = sum(1 for meta in all_docs['metadatas'] if '1068' in meta.get('source', ''))
print(f"✓ {conges_count} chunks du fichier congés trouvés")

if conges_count > 0:
    for idx, meta in enumerate(all_docs['metadatas']):
        if '1068' in meta.get('source', ''):
            print(f"\nChunk trouvé:")
            print(f"  Source: {meta.get('source', 'N/A')}")
            print(f"  Contenu: {all_docs['documents'][idx][:300]}")
            break
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "✓ DIAGNOSTIC TERMINÉ"
echo "=========================================="
