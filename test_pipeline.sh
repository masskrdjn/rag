#!/bin/bash
# Script pour tester le pipeline RAG complet

echo "Test du Pipeline RAG"
echo "===================="

sudo -u ragapp bash << 'EOF'
cd /home/ragapp/rag-system
source venv/bin/activate

python3 << 'PYTHON_SCRIPT'
from rag_pipeline import SimpleRAG
import time

print("\n1. Initialisation du RAG...")
rag = SimpleRAG()
rag.setup_chain()
print("✓ RAG configuré")

print("\n2. Test question: 'quelles sont les regles des conges'")
start = time.time()
answer = rag.ask("quelles sont les regles des conges")
elapsed = time.time() - start

print(f"\nRéponse (en {elapsed:.2f}s):")
print(answer)

print("\n3. Test question: 'congés payés'")
start = time.time()
answer = rag.ask("congés payés")
elapsed = time.time() - start

print(f"\nRéponse (en {elapsed:.2f}s):")
print(answer)

print("\n4. Test avec contexte explicite")
start = time.time()
answer = rag.ask("Quelles sont les règles concernant les congés payés et absences ?")
elapsed = time.time() - start

print(f"\nRéponse (en {elapsed:.2f}s):")
print(answer)
PYTHON_SCRIPT

EOF
