#!/usr/bin/env python3
"""Debug du retrieval pour la question sur les congés payés"""
import sys
# sys.path.insert(0, '/home/rag')  # Pas nécessaire avec installation globale

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

print("="*80)
print("DEBUG RETRIEVAL - Question Congés Payés")
print("="*80)

# Initialize
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(
    persist_directory="/home/rag/chroma_db",
    embedding_function=embeddings
)

question = "Comment faire une demande de congés payés ?"
print(f"\n📝 Question: {question}")

# Test différentes configurations
configs = [
    ("Similarity k=3", {"search_type": "similarity", "search_kwargs": {"k": 3}}),
    ("Similarity k=5", {"search_type": "similarity", "search_kwargs": {"k": 5}}),
    ("Similarity k=8", {"search_type": "similarity", "search_kwargs": {"k": 8}}),
    ("Threshold 0.7", {"search_type": "similarity_score_threshold", "search_kwargs": {"k": 8, "score_threshold": 0.7}}),
    ("Threshold 0.5", {"search_type": "similarity_score_threshold", "search_kwargs": {"k": 8, "score_threshold": 0.5}}),
]

for config_name, config_params in configs:
    print(f"\n{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")
    
    retriever = vectorstore.as_retriever(**config_params)
    docs = retriever.invoke(question)
    
    print(f"\n📊 Nombre de documents récupérés: {len(docs)}")
    
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        
        # Afficher les métadonnées
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', 'N/A')
            print(f"Source: {source}")
            
            # Extraire le nom du fichier
            if '/' in source:
                filename = source.split('/')[-1]
                print(f"Fichier: {filename}")
        
        # Afficher un extrait du contenu
        content = doc.page_content
        print(f"Contenu ({len(content)} chars):")
        print(content[:300] + "..." if len(content) > 300 else content)
        
        # Vérifier si c'est le bon document
        if "congés" in content.lower() and "whaller" in content.lower():
            print("✅ DOCUMENT CORRECT TROUVÉ!")
        elif "congés" in content.lower():
            print("⚠️  Contient 'congés' mais pas 'Whaller'")
        else:
            print("❌ PAS LE BON DOCUMENT")

print(f"\n{'='*80}")
print("Recherche directe du document sur les congés...")
print(f"{'='*80}")

# Recherche avec des mots-clés spécifiques
all_docs = vectorstore.similarity_search("Whaller Penguin World congés payés absences", k=10)
print(f"\n📊 Documents trouvés avec 'Whaller Penguin World congés': {len(all_docs)}")

for i, doc in enumerate(all_docs, 1):
    content = doc.page_content.lower()
    if "whaller" in content and "penguin" in content:
        print(f"\n✅ Document {i} contient 'Whaller' et 'Penguin':")
        source = doc.metadata.get('source', 'N/A') if hasattr(doc, 'metadata') else 'N/A'
        print(f"   Source: {source}")
        print(f"   Extrait: {doc.page_content[:200]}")
        break
else:
    print("❌ Aucun document avec 'Whaller' et 'Penguin' trouvé dans le top 10")
