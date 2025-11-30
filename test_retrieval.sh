#!/bin/bash
# Script pour tester la recherche dans ChromaDB

echo "Test de recherche dans ChromaDB"
echo "================================"

cd /home/rag

python3 << 'PYTHON_SCRIPT'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("\n1. Chargement de la base...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="/home/rag/chroma_db",
    embedding_function=embeddings
)

print(f"Total documents: {vectorstore._collection.count()}")

print("\n2. Test recherche: 'congés payés'")
results = vectorstore.similarity_search_with_score("congés payés", k=3)
print(f"Trouvé {len(results)} résultats\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"--- Résultat {i} (Score: {score:.4f}) ---")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Contenu: {doc.page_content[:300]}...")
    print()

print("\n3. Test recherche: 'quelles sont les regles des conges'")
results = vectorstore.similarity_search_with_score("quelles sont les regles des conges", k=3)
print(f"Trouvé {len(results)} résultats\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"--- Résultat {i} (Score: {score:.4f}) ---")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Contenu: {doc.page_content[:300]}...")
    print()
PYTHON_SCRIPT
