#!/usr/bin/env python3
"""Script de diagnostic de la base ChromaDB"""
import sys
# sys.path.insert(0, '/home/rag')  # Pas nécessaire avec installation globale

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

print("="*60)
print("DIAGNOSTIC BASE CHROMADB")
print("="*60)

try:
    print("\n1. Chargement de la base...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="/home/rag/chroma_db",
        embedding_function=embeddings
    )
    
    print(f"✓ Base chargée")
    
    print("\n2. Comptage des documents...")
    count = vectorstore._collection.count()
    print(f"✓ Total: {count} chunks dans la base")
    
    if count == 0:
        print("\n⚠️  LA BASE EST VIDE!")
        sys.exit(1)
    
    print("\n3. Test de recherche: 'congés payés'")
    results = vectorstore.similarity_search_with_score("congés payés", k=5)
    
    print(f"✓ Trouvé {len(results)} résultats\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Résultat {i} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Contenu: {doc.page_content[:300]}")
        print()
    
    print("\n4. Test de recherche: 'regles conges absences'")
    results = vectorstore.similarity_search_with_score("regles conges absences", k=3)
    
    print(f"✓ Trouvé {len(results)} résultats\n")
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Résultat {i} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Contenu: {doc.page_content[:300]}")
        print()
    
    print("\n5. Recherche du fichier spécifique '1068_Conges'")
    all_docs = vectorstore.get()
    conges_docs = [
        (id, meta) for id, meta in zip(all_docs['ids'], all_docs['metadatas'])
        if '1068' in meta.get('source', '') or 'Conges' in meta.get('source', '')
    ]
    
    print(f"✓ Trouvé {len(conges_docs)} chunks du fichier congés")
    if conges_docs:
        print(f"  Exemple: {conges_docs[0][1].get('source', 'N/A')}")
        # Afficher le contenu
        for doc_id, _ in conges_docs[:2]:
            idx = all_docs['ids'].index(doc_id)
            print(f"\n  Contenu du chunk {doc_id[:8]}...")
            print(f"  {all_docs['documents'][idx][:400]}")
    
    print("\n" + "="*60)
    print("✓ DIAGNOSTIC BASE TERMINÉ")
    print("="*60)
    
except Exception as e:
    print(f"\n✗ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
