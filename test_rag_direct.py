#!/usr/bin/env python3
"""Test direct du pipeline RAG (sans serveur)"""
import sys
sys.path.insert(0, '/home/ragapp/rag-system/venv/lib/python3.10/site-packages')

from rag_pipeline import SimpleRAG
import time

print("="*60)
print("TEST DIRECT PIPELINE RAG")
print("="*60)

print("\n1. Initialisation du RAG...")
try:
    rag = SimpleRAG()
    rag.persist_directory = "/home/ragapp/rag-system/chroma_db"
    print("✓ RAG initialisé")
except Exception as e:
    print(f"✗ Erreur initialisation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Configuration de la chaîne...")
try:
    start = time.time()
    rag.setup_chain()
    elapsed = time.time() - start
    print(f"✓ Chaîne configurée en {elapsed:.2f}s")
except Exception as e:
    print(f"✗ Erreur configuration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Test d'une question...")
question = "quelles sont les regles des conges"
try:
    print(f"Question: '{question}'")
    start = time.time()
    answer = rag.ask(question)
    elapsed = time.time() - start
    print(f"✓ Réponse reçue en {elapsed:.2f}s")
    print(f"\nRéponse: {answer}")
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("FIN DU TEST")
print("="*60)
