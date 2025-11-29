#!/usr/bin/env python3
import time
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("Début du test de chargement ChromaDB...")
start = time.time()

print("1. Initialisation des embeddings...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
print(f"   ✓ Fait en {time.time() - start:.2f}s")

print("2. Chargement de la base ChromaDB...")
step_start = time.time()
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
print(f"   ✓ Fait en {time.time() - step_start:.2f}s")

print("3. Comptage des documents...")
step_start = time.time()
count = vectorstore._collection.count()
print(f"   ✓ {count} documents - Fait en {time.time() - step_start:.2f}s")

print("4. Test de recherche...")
step_start = time.time()
results = vectorstore.similarity_search("congés payés", k=3)
print(f"   ✓ {len(results)} résultats - Fait en {time.time() - step_start:.2f}s")

print(f"\n✓ Total: {time.time() - start:.2f}s")

if results:
    print(f"\nPremier résultat:")
    print(f"  Source: {results[0].metadata.get('source', 'N/A')}")
    print(f"  Contenu: {results[0].page_content[:200]}...")
