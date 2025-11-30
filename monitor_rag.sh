#!/bin/bash
# Script pour surveiller et diagnostiquer l'état du RAG

# PROCÉDURE DE DIAGNOSTIC QUAND LE RAG NE RÉPOND PAS
# ===================================================
# 
# Symptômes : Une requête à http://localhost:8000/ask prend trop de temps
#
# 1. Vérifier l'état du processus Ollama :
#    ps aux | grep '[o]llama'
#    → Si un processus "ollama runner" utilise >300% CPU depuis longtemps, il est bloqué
#
# 2. Arrêter le processus bloqué :
#    sudo pkill -9 -f 'ollama runner'
#
# 3. Vérifier les logs du serveur :
#    tail -100 /home/rag/server.log
#
# 4. Causes fréquentes :
#    - Contexte trop long envoyé au LLM (trop de documents récupérés)
#    - Modèle LLM trop lourd pour la machine (llama3.2:3.2B)
#    - Prompt trop complexe générant une réponse trop longue
#
# 5. Solutions :
#    - Réduire top_k à 3 dans rag_pipeline.py
#    - Utiliser un modèle plus léger : ollama pull llama3.2:1b
#    - Ajouter un timeout dans server.py
#    - Utiliser un GPU si disponible

echo "=== SURVEILLANCE DU RAG ==="
echo ""

echo "1. État du serveur :"
ps aux | grep -E '(uvicorn|server\.py)' | grep -v grep
echo ""

echo "2. État d'Ollama (vérifier CPU%) :"
ps aux | grep '[o]llama' | head -5
echo ""

echo "3. Connexions actives sur le port 8000 :"
sudo lsof -i :8000 2>/dev/null || echo "Aucune connexion ou lsof non disponible"
echo ""

echo "4. Derniers logs du serveur (10 lignes) :"
tail -10 /home/rag/server.log 2>/dev/null || echo "Logs non accessibles"
echo ""

echo "5. Test de santé de l'API :"
curl -s http://localhost:8000/health
echo ""

echo ""
echo "=== ACTIONS RAPIDES ==="
echo "Pour arrêter un processus Ollama bloqué :"
echo "  sudo pkill -9 -f 'ollama runner'"
echo ""
echo "Pour redémarrer le serveur :"
echo "  ./restart_server.sh"
echo ""
