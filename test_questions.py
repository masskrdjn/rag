#!/usr/bin/env python3
"""Test API avec différentes formulations"""
import requests
import json

def test_question(question):
    print(f"\nQuestion: '{question}'")
    print("-" * 60)
    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question},
            timeout=120
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Réponse: {result['answer'][:500]}")
        else:
            print(f"Erreur {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Erreur: {e}")

print("="*60)
print("TEST DE DIFFÉRENTES FORMULATIONS")
print("="*60)

test_question("congés payés")
test_question("congés")
test_question("absences")
test_question("Whaller Penguin World congés")
test_question("règles entreprise congés absences")

print("\n" + "="*60)
