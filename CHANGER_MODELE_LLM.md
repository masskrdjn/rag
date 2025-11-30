# Procédure de changement de modèle LLM

Ce document décrit la procédure complète et réutilisable pour charger un nouveau modèle LLM dans le système RAG et basculer sur celui-ci.

## Prérequis

- Accès SSH au serveur
- Droits sudo (pour systemctl si déployé en service)
- Ollama installé et fonctionnel

---

## Étape 1 : Vérifier les modèles disponibles

```bash
# Lister les modèles déjà installés
ollama list

# Exemple de sortie :
# NAME              SIZE      MODIFIED
# llama3.2:latest   2.0 GB    1 day ago
# nomic-embed-text  274 MB    1 day ago
```

---

## Étape 2 : Télécharger le nouveau modèle

```bash
# Télécharger le modèle souhaité (exemple : mistral 7B)
ollama pull mistral:7b

# Autres modèles courants :
# ollama pull mistral:latest        # Dernière version Mistral
# ollama pull mistral:7b-instruct   # Version instruction-tuned
# ollama pull llama3.2              # Llama 3.2
# ollama pull mixtral:8x7b          # Mixtral MoE (plus lourd)
# ollama pull codellama:7b          # Pour le code
# ollama pull phi3:medium           # Microsoft Phi-3
```

**Note :** Le téléchargement peut prendre plusieurs minutes selon la taille du modèle et la connexion.

---

## Étape 3 : Tester le modèle manuellement (optionnel mais recommandé)

```bash
# Tester que le modèle fonctionne correctement
ollama run mistral:7b "Dis bonjour en français"

# Vérifier les performances de base
time ollama run mistral:7b "Quelle est la capitale de la France ?"
```

---

## Étape 4 : Modifier la configuration du RAG

### Option A : Modification directe du fichier (recommandée pour changement permanent)

Éditer le fichier `/home/rag/rag_pipeline.py` :

```bash
# Ouvrir le fichier
nano /home/rag/rag_pipeline.py
# ou
vim /home/rag/rag_pipeline.py
```

Localiser la ligne (vers le début de la classe `SimpleRAG`) :

```python
self.model_name = "llama3.2"
```

Remplacer par le nouveau modèle :

```python
self.model_name = "mistral:7b"
```

**Sauvegarder le fichier.**

### Option B : Via variable d'environnement (pour tests rapides)

Modifier `/home/rag/rag_pipeline.py` pour supporter une variable d'environnement :

```python
# Dans __init__ de SimpleRAG, remplacer :
self.model_name = "llama3.2"

# Par :
self.model_name = os.environ.get("RAG_LLM_MODEL", "llama3.2")
```

Puis lancer avec :

```bash
export RAG_LLM_MODEL="mistral:7b"
python3 server.py
```

---

## Étape 5 : Redémarrer le serveur RAG

### Si lancé manuellement (mode développement)

```bash
# Utiliser le script de redémarrage
cd /home/rag
./restart_server.sh

# OU redémarrer manuellement
pkill -f 'python3 server.py'
sleep 2
cd /home/rag && nohup python3 server.py > server.log 2>&1 &
```

### Si déployé en service systemd (mode production)

```bash
# Redémarrer le service
sudo systemctl restart rag-api

# Vérifier le statut
sudo systemctl status rag-api

# Consulter les logs si besoin
sudo journalctl -u rag-api -f --no-pager -n 50
```

---

## Étape 6 : Vérifier le fonctionnement

### Test de santé

```bash
curl http://localhost:8000/health
# Réponse attendue : {"status":"healthy"}
```

### Test fonctionnel

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Bonjour, peux-tu te présenter ?"}'
```

### Vérifier les logs du serveur

```bash
# Logs en temps réel
tail -f /home/rag/server.log
```

---

## Récapitulatif des commandes (copier-coller rapide)

```bash
# === CHANGEMENT DE MODÈLE - EXEMPLE MISTRAL 7B ===

# 1. Télécharger le modèle
ollama pull mistral:7b

# 2. Modifier le fichier de configuration
sed -i 's/self.model_name = ".*"/self.model_name = "mistral:7b"/' /home/rag/rag_pipeline.py

# 3. Vérifier la modification
grep "model_name" /home/rag/rag_pipeline.py

# 4. Redémarrer le serveur
cd /home/rag && ./restart_server.sh

# 5. Tester
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"Test rapide"}'
```

---

## Rollback (retour en arrière)

En cas de problème, revenir au modèle précédent :

```bash
# Remettre l'ancien modèle
sed -i 's/self.model_name = ".*"/self.model_name = "llama3.2"/' /home/rag/rag_pipeline.py

# Redémarrer
cd /home/rag && ./restart_server.sh
```

---

## Modèles recommandés par usage

| Modèle | Taille | Usage recommandé | Commande |
|--------|--------|------------------|----------|
| `mistral:7b` | ~4 GB | Usage général, bon équilibre | `ollama pull mistral:7b` |
| `mistral:7b-instruct` | ~4 GB | Suivi d'instructions | `ollama pull mistral:7b-instruct` |
| `llama3.2` | ~2 GB | Léger, rapide | `ollama pull llama3.2` |
| `llama3.2:3b` | ~2 GB | Très léger | `ollama pull llama3.2:3b` |
| `mixtral:8x7b` | ~26 GB | Haute qualité (GPU requis) | `ollama pull mixtral:8x7b` |
| `phi3:medium` | ~7 GB | Bon raisonnement | `ollama pull phi3:medium` |
| `qwen2:7b` | ~4 GB | Multilingue | `ollama pull qwen2:7b` |

---

## Dépannage

### Erreur "model not found"

```bash
# Vérifier que le modèle est bien installé
ollama list

# Retélécharger si nécessaire
ollama pull mistral:7b
```

### Serveur ne démarre pas

```bash
# Consulter les logs
tail -100 /home/rag/server.log

# Vérifier la syntaxe Python
python3 -m py_compile /home/rag/rag_pipeline.py
```

### Ollama ne répond pas

```bash
# Vérifier le service Ollama
sudo systemctl status ollama

# Redémarrer Ollama si nécessaire
sudo systemctl restart ollama
```

### Réponses lentes

```bash
# Vérifier l'utilisation mémoire
free -h

# Vérifier si le modèle est chargé
curl http://localhost:11434/api/tags
```

---

## Script automatisé

Un script `change_model.sh` est fourni pour automatiser tout le processus :

```bash
./change_model.sh mistral:7b
```

Voir le fichier `/home/rag/change_model.sh` pour plus de détails.
