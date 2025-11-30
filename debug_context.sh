#!/bin/bash
# Script pour voir exactement ce que le LLM reçoit comme contexte

echo "Debug du contexte envoyé au LLM"
echo "==============================="

cd /home/rag

python3 << 'PYTHON_SCRIPT'
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

print("\n1. Initialisation...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="/home/rag/chroma_db",
    embedding_function=embeddings
)

# Test avec différentes questions
questions = [
    "congés payés",
    "quelles sont les regles des conges",
    "règles entreprise congés absences"
]

for question in questions:
    print(f"\n{'='*70}")
    print(f"QUESTION: '{question}'")
    print('='*70)
    
    # Récupérer les documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    
    print(f"\n{len(docs)} documents récupérés:\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Contenu complet:")
        print(doc.page_content)
        print()
    
    # Formatter comme dans le RAG
    context = "\n\n".join(doc.page_content for doc in docs)
    
    print(f"\n{'='*70}")
    print("CONTEXTE TOTAL ENVOYÉ AU LLM:")
    print('='*70)
    print(context)
    print(f"\n{'='*70}")
    print(f"Longueur du contexte: {len(context)} caractères")
    print('='*70)

PYTHON_SCRIPT
