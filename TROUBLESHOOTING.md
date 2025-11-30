# 🔧 Guide de Troubleshooting - RAG API

## Problèmes courants et solutions

---

## 1. ❌ La réponse ne correspond pas aux modifications effectuées

### Symptômes
- Les `content_preview` dans la réponse montrent l'ancien format
- Les améliorations du pipeline ne sont pas visibles
- Résultats incohérents entre tests directs et API

### Cause
Le serveur utilise une ancienne instance avec un cache mémoire de la base vectorielle.

### Solution
```bash
# Redémarrer proprement le serveur
./restart_rag.sh
```

Ou manuellement :
```bash
# 1. Tuer tous les processus existants
pkill -9 -f "python3 server.py"
pkill -9 -f "uvicorn"
fuser -k 8000/tcp 2>/dev/null

# 2. Attendre
sleep 2

# 3. Redémarrer le serveur
cd /home/rag && python3 server.py &

# 4. Vérifier
sleep 10 && ss -tlnp | grep 8000
```

---

## 2. ❌ Le serveur ne démarre pas / Port 8000 non accessible

### Symptômes
- `ss -tlnp | grep 8000` ne retourne rien
- Erreur "Connection refused" dans Postman

### Diagnostic
```bash
# Vérifier si un processus bloque le port
fuser 8000/tcp

# Vérifier les processus Python
ps aux | grep -E "python|uvicorn" | grep -v grep

# Voir les erreurs du serveur
cd /home/rag && python3 server.py
```

### Solutions possibles

#### Port déjà utilisé
```bash
fuser -k 8000/tcp
```

#### Erreur d'import Python
```bash
cd /home/rag && python3 -c "from rag_pipeline import SimpleRAG; print('OK')"
```

#### Dépendances manquantes
```bash
pip3 install fastapi uvicorn langchain-chroma langchain-ollama langchain-community
```

---

## 3. ❌ Erreur "Field required" ou body vide

### Symptômes
```json
{"detail":[{"type":"missing","loc":["body"],"msg":"Field required"}]}
```

### Cause
Le serveur n'était pas complètement démarré ou la requête est mal formée.

### Solution
1. Attendre 10-15 secondes après le démarrage du serveur
2. Vérifier le format de la requête :
```json
{
  "question": "Votre question ici"
}
```

---

## 4. ❌ Résultats de mauvaise qualité après réingestion

### Symptômes
- Les chunks ne contiennent pas le bon contexte
- Le sommaire n'apparaît pas dans les sources

### Solution
```bash
# 1. Supprimer la base existante
rm -rf /home/rag/chroma_db

# 2. Réingérer
cd /home/rag && python3 ingest_html_adaptive.py

# 3. Redémarrer le serveur
./restart_rag.sh
```

---

## 5. ❌ Problèmes d'encodage (caractères Ã©, Ã , etc.)

### Symptômes
- Caractères mal affichés : `Ã©` au lieu de `é`

### Cause
Problème d'encodage UTF-8 dans PowerShell ou le client HTTP.

### Solution
Dans PowerShell :
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$response = Invoke-RestMethod -Uri "http://localhost:8000/ask" ...
```

Ou utiliser WSL directement :
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Votre question"}' | python3 -m json.tool
```

---

## 📋 Commandes de diagnostic rapide

```bash
# État du serveur
./diagnose_rag.sh

# Redémarrage complet
./restart_rag.sh

# Test rapide
curl -s http://localhost:8000/health
```

---

## 🔄 Procédure de redémarrage complète

Si rien ne fonctionne, suivre cette procédure complète :

```bash
# 1. Arrêter tout
pkill -9 -f python3
fuser -k 8000/tcp 2>/dev/null

# 2. Vérifier la base de données
ls -la /home/rag/chroma_db/

# 3. Si nécessaire, réingérer
python3 /home/rag/ingest_html_adaptive.py

# 4. Démarrer le serveur
cd /home/rag && python3 server.py &

# 5. Attendre et tester
sleep 15
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}' | head -c 200
```

---

## 📞 Informations utiles

| Élément | Valeur |
|---------|--------|
| Port API | 8000 |
| Base ChromaDB | `/home/rag/chroma_db` |
| Modèle LLM | llama3.2 |
| Modèle Embedding | nomic-embed-text |
| Fichier serveur | `/home/rag/server.py` |
| Pipeline RAG | `/home/rag/rag_pipeline.py` |
| Script ingestion | `/home/rag/ingest_html_adaptive.py` |
