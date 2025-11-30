#!/bin/bash
# Script pour debug le retriever

echo "Debug du Retriever"
echo "=================="

cd /home/rag

python3 << 'PYTHON_SCRIPT'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("\n1. Initialisation...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="/home/rag/chroma_db",
    embedding_function=embeddings
)

print("✓ Vectorstore créé")

# Créer un retriever avec différentes configurations
print("\n2. Test retriever par défaut")
retriever = vectorstore.as_retriever()
docs = retriever.invoke("quelles sont les regles des conges")
print(f"Trouvé {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    print(f"\n  Doc {i}:")
    print(f"  Source: {doc.metadata.get('source', 'N/A')}")
    print(f"  Contenu: {doc.page_content[:200]}...")

print("\n3. Test retriever avec k=5")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("congés payés")
print(f"Trouvé {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    print(f"\n  Doc {i}:")
    print(f"Source: {doc.metadata.get('source', 'N/A')}")
    print(f"Contenu: {doc.page_content[:200]}...")

PYTHON_SCRIPT
