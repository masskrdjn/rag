#!/usr/bin/env python3
"""Test de la réponse améliorée sur la question des congés"""
import sys
sys.path.insert(0, '/home/ragapp/rag-system/venv/lib/python3.10/site-packages')

from rag_pipeline import SimpleRAG

print("="*80)
print("TEST - Réponse sur les congés avec différentes configurations")
print("="*80)

question = "Comment faire une demande de congés payés ?"

# Test 1: k=3 (moins de bruit)
print("\n" + "="*80)
print("Configuration 1: k=3 (moins de documents, moins de bruit)")
print("="*80)
rag1 = SimpleRAG(retrieval_mode="similarity", top_k=3)
rag1.persist_directory = "/home/ragapp/rag-system/chroma_db"
print(f"\nQuestion: {question}\n")
response1 = rag1.ask(question)
print(f"Réponse:\n{response1}")

# Test 2: k=1 (seulement le meilleur document)
print("\n" + "="*80)
print("Configuration 2: k=1 (uniquement le document le plus pertinent)")
print("="*80)
rag2 = SimpleRAG(retrieval_mode="similarity", top_k=1)
rag2.persist_directory = "/home/ragapp/rag-system/chroma_db"
print(f"\nQuestion: {question}\n")
response2 = rag2.ask(question)
print(f"Réponse:\n{response2}")

# Test 3: Score threshold strict
print("\n" + "="*80)
print("Configuration 3: Threshold 0.8 (très strict)")
print("="*80)
rag3 = SimpleRAG(retrieval_mode="similarity_score_threshold", top_k=5, score_threshold=0.8)
rag3.persist_directory = "/home/ragapp/rag-system/chroma_db"
print(f"\nQuestion: {question}\n")
try:
    response3 = rag3.ask(question)
    print(f"Réponse:\n{response3}")
except Exception as e:
    print(f"Erreur: {e}")

print("\n" + "="*80)
print("Analyse: Quelle configuration donne la meilleure réponse?")
print("="*80)
print("✅ La réponse attendue devrait mentionner:")
print("   - Consulter la sphère Whaller Penguin World")
print("   - Sa box de sphère")
print("   - PAS de procédures de cotation ou compagnies aériennes")
