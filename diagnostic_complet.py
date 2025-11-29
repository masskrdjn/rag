#!/usr/bin/env python3
"""Script de diagnostic complet du RAG"""
import sys
import subprocess
import time

print("="*60)
print("DIAGNOSTIC COMPLET DU SYSTÈME RAG")
print("="*60)

# 0. Vérification du serveur
print("\n0. VÉRIFICATION DU SERVEUR")
print("-" * 60)
try:
    result = subprocess.run(
        ["ps", "aux"],
        capture_output=True,
        text=True
    )
    uvicorn_processes = [line for line in result.stdout.split('\n') if 'uvicorn' in line and 'server:app' in line]
    if uvicorn_processes:
        print(f"✓ Serveur en cours d'exécution ({len(uvicorn_processes)} processus)")
        for proc in uvicorn_processes:
            print(f"  {proc[:100]}")
    else:
        print("✗ Serveur non démarré")
        print("\nRedémarrage du serveur...")
        subprocess.run([
            "sudo", "su", "-", "ragapp", "-c",
            "cd /home/ragapp/rag-system && nohup venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &"
        ])
        print("Attente de 10 secondes...")
        time.sleep(10)
except Exception as e:
    print(f"Erreur: {e}")

# 1. Test de l'API
print("\n1. TEST DE L'API")
print("-" * 60)

import requests
import json

try:
    print("Test /health...")
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✓ API accessible - Status: {response.status_code}")
    print(f"  Réponse: {response.json()}")
except Exception as e:
    print(f"✗ API inaccessible: {e}")
    print("\n⚠️  Le serveur ne répond toujours pas.")
    
    # Afficher les logs
    print("\nDernières lignes du log serveur:")
    try:
        result = subprocess.run(
            ["sudo", "tail", "-20", "/home/ragapp/rag-system/server.log"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except:
        pass
    sys.exit(1)

# 2. Test de recherche simple
print("\n2. TEST DE RECHERCHE SIMPLE")
print("-" * 60)
question = "quelles sont les regles des conges"
try:
    print(f"Question: '{question}'")
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question},
        timeout=30
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nRéponse:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"✗ Erreur: {e}")

# 3. Test avec une question plus précise
print("\n3. TEST AVEC QUESTION PRÉCISE")
print("-" * 60)
question = "Congés payés et absences"
try:
    print(f"Question: '{question}'")
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question},
        timeout=30
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nRéponse:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"✗ Erreur: {e}")

# 4. Test avec mention explicite de Whaller
print("\n4. TEST RECHERCHE WHALLER")
print("-" * 60)
question = "Ou trouver les informations sur les conges dans Whaller"
try:
    print(f"Question: '{question}'")
    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question},
        timeout=30
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"\nRéponse:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"✗ Erreur: {e}")

print("\n" + "="*60)
print("FIN DU DIAGNOSTIC")
print("="*60)
