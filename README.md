# RAG Enterprise — API locale (FastAPI + Ollama + ChromaDB)

Pipeline RAG en français pour interroger une base documentaire HTML d'entreprise
(procédures GDS, RH, voyages…). Tout tourne en local : embeddings et LLM via
Ollama, base vectorielle ChromaDB persistée sur disque, API FastAPI.

## Architecture

```
        ┌─────────────┐
HTTP →  │  server.py  │  FastAPI : /health, /ask
        └──────┬──────┘
               │
        ┌──────▼─────────┐    ┌─────────────────────┐
        │ rag_pipeline.py│───►│ ChromaDB (vector)   │
        │  - retrieval   │    │ + BM25 (en mémoire) │
        │  - rerank      │    └─────────────────────┘
        │  - expansion   │
        │  - anti-halluc │    ┌─────────────────────┐
        │  - cache       │───►│ Ollama (embed+LLM)  │
        └────────────────┘    └─────────────────────┘
```

Modules d'optimisation :

| Fichier | Rôle |
| --- | --- |
| `reranker.py` / `reranker_gpu.py` | Reranking BM25/keyword (CPU) ou CrossEncoder (GPU). |
| `query_expander.py` | Expansion : synonymes métier + morphologie + LLM (optionnel). |
| `hallucination_detector.py` / `_gpu.py` | TF-IDF + word-overlap pour détecter les claims non sourcés. |
| `cache_manager.py` | Cache SQLite des réponses validées. |
| `device_config.py` | Détection CPU/GPU pilotée par `RAG_USE_GPU`. |

## Prérequis

- Python ≥ 3.10
- [Ollama](https://ollama.com/) installé et démarré, avec au minimum :
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:14b   # ou un autre modèle de config.py
  ```
- ~10 Go d'espace disque pour les modèles + ChromaDB.
- (Optionnel) GPU NVIDIA avec PyTorch + CUDA pour activer les variantes GPU.

## Installation

```bash
pip install -r requirements.txt
```

Pour la pile GPU optionnelle, dé-commenter `torch` et `sentence-transformers` dans
`requirements.txt`, puis :

```bash
export RAG_USE_GPU=1
```

## Configuration

Tout passe par `config.py` ou par variables d'environnement (qui surchargent) :

| Variable | Défaut | Effet |
| --- | --- | --- |
| `RAG_ACTIVE_MODEL` | `qwen-14b` | Clé du modèle dans `config.MODELS`. |
| `RAG_MODEL` | — | Nom Ollama brut, prioritaire (ex. `mistral:7b`). |
| `RAG_EMBEDDING_MODEL` | `nomic-embed-text` | Modèle d'embeddings Ollama. |
| `RAG_CHROMA_DB_PATH` | `./chroma_db` (Win) ou `/home/rag/chroma_db` (Linux si dispo) | Dossier de persistance ChromaDB. |
| `RAG_DATA_PATH` | `./data` | Dossier source HTML pour l'ingestion. |
| `RAG_TOP_K` | `6` | Nombre de candidats avant reranking. |
| `RAG_MAX_DYNAMIC_TOP_K` | `8` | Plafond du `top_k` dynamique selon la question. |
| `RAG_MAX_QUESTION_CHARS` | `1000` | Taille maximale acceptee par `/ask`. |
| `RAG_MAX_CONTEXT` | `3000` | Longueur max (chars) du contexte injecté au LLM. |
| `RAG_USE_HYBRID` | `1` | Active la branche BM25 hybride. |
| `RAG_CACHE_TTL_SECONDS` | â€” | TTL optionnel du cache SQLite. Vide = pas d'expiration temporelle. |
| `RAG_CORPUS_BUILD_ID` | hash leger de Chroma | Identifiant de corpus pour invalider le cache apres reingestion. |
| `RAG_API_KEY` | â€” | Si defini, `/ask` exige le header `X-API-Key`. |
| `RAG_KEYWORDS_CONFIG` | â€” | Fichier JSON optionnel pour remplacer/etendre les mots-cles retrieval. |
| `RAG_USE_GPU` | `auto` | `0`/`1`/`auto` pour le placement device. `1` autorise CUDA côté API si `RAG_DISABLE_CUDA` n'est pas défini. |
| `RAG_DISABLE_CUDA` | `1` côté API | Si `1`, désactive CUDA côté API uniquement (compat. vieux GPU). Mettre `0` pour l'autoriser explicitement. |

Lister les modèles disponibles :

```bash
python config.py
```

## Workflow type

### 1. Ingestion

Placer les fichiers `.html` dans `data/` (ou pointer `RAG_DATA_PATH` ailleurs),
puis :

```bash
python ingest_html_adaptive.py
```

Le chunking est adaptatif : sections `+-Titre` détectées (Amadeus / process),
chunk size variable selon la longueur du document, métadonnées enrichies
(catégories, signaux de fiabilité, etc.).

Sous Linux, le script `reingest_adaptive.sh` fait sauvegarde + ingestion +
restauration automatique en cas d'échec.

### 2. Démarrage de l'API

```bash
python server.py
# ou
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

Sous Linux (hors systemd), `restart.sh` arrête les processus existants, vérifie
que la base ChromaDB existe, relance le serveur et attend `/health`. Il
respecte `RAG_CHROMA_DB_PATH`, `PROJECT_DIR`, `RAG_PORT` et `RAG_LOG_FILE`.

Endpoints :

- `GET /health` → `{"status": "healthy"}`
- `POST /ask` → `{ answer, sources[], metadata }`

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quelles sont les règles des congés ?"}'
```

### 3. Tests

```bash
python test_api_direct.py     # health + une question
python test_rag.py            # batch sur test_questions.json, log fichier
python diagnostic_db.py       # inspection ChromaDB
python clear_cache.py         # purge le cache des réponses
```

## Déploiement Linux (systemd)

`rag-api.service` est fourni à titre d'exemple ; adapter `User`,
`WorkingDirectory` et `Environment` :

```bash
sudo cp rag-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rag-api.service
journalctl -u rag-api.service -f
```

## Structure du dépôt

```
.
├── server.py                         # API FastAPI
├── rag_pipeline.py                   # Pipeline principal
├── config.py                         # Configuration centrale
├── device_config.py                  # Détection CPU/GPU
├── ingest_html_adaptive.py           # Ingestion HTML adaptative
├── reranker.py / reranker_gpu.py
├── query_expander.py
├── hallucination_detector.py / _gpu.py
├── cache_manager.py
├── clear_cache.py                    # purge cache
├── diagnostic_db.py                  # diag ChromaDB
├── test_api_direct.py                # smoke test API
├── test_rag.py                       # batch test
├── test_questions.json               # questions métier
├── reingest_adaptive.sh              # réingestion (Linux)
├── restart.sh                        # redémarrage serveur (Linux)
├── rag-api.service                   # unit systemd (exemple)
├── requirements.txt
├── AGENTS.md                         # consignes pour agents IA
└── README.md
```

## Benchmark Embeddings

`bge-m3` est documente comme candidat prioritaire pour le francais, mais il
necessite une reingestion dans une base Chroma dediee avant comparaison :

```bash
RAG_EMBEDDING_MODEL=nomic-embed-text RAG_CHROMA_DB_PATH=./chroma_nomic python ingest_html_adaptive.py
RAG_EMBEDDING_MODEL=bge-m3 RAG_CHROMA_DB_PATH=./chroma_bge_m3 python ingest_html_adaptive.py
python benchmark_embeddings.py --embedding nomic-embed-text --chroma ./chroma_nomic --embedding bge-m3 --chroma ./chroma_bge_m3
```

## Notes

- Le pipeline n'autorise en cache que les réponses dont la confiance
  anti-hallucination dépasse `0.5` (cf. `rag_pipeline.SimpleRAG.ask`).
- `_estimate_dynamic_topk` renvoie 4 sources sur les questions précises
  (mots-clés métier détectés), 8 sur les questions larges/procédurales,
  6 par défaut.
- L'expansion LLM n'est déclenchée que sur les questions vraiment vagues
  (ex. « que faire si l'émission échoue ? »). Les questions ciblées
  (« couleurs du robot », « règles des congés ») la court-circuitent.
- En production Linux, vérifier que `RAG_CHROMA_DB_PATH` pointe bien vers la
  base ingérée. Le défaut Linux (`/home/rag/chroma_db`) est conservé pour
  rester compatible avec l'historique du dépôt.
