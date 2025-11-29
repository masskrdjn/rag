#!/usr/bin/env python3
"""Test direct de l'API avec timeout adapté"""
import requests
import json
import sys

print("="*60)
print("TEST API RAG")
print("="*60)

# Test 1: Health check
print("\n1. Test /health...")
try:
    response = requests.get("http://localhost:8000/health", timeout=30)
    print(f"✓ Status: {response.status_code}")
    print(f"  Réponse: {response.json()}")
except Exception as e:
    print(f"✗ Erreur: {e}")
    sys.exit(1)

# Test 2: Question simple
print("\n2. Test question simple...")
question = "quelles sont les regles des conges"
try:
    print(f"Question: '{question}'")
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question},
        timeout=120  # 2 minutes pour laisser le temps
    )
    print(f"✓ Status: {response.status_code}")
    result = response.json()
    print(f"\nRéponse:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"✗ Erreur: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("FIN DU TEST")
print("="*60)
