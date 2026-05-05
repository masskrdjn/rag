# AGENTS.md

Guide de travail pour les agents intervenant sur ce projet RAG.

## Vue D'ensemble

Ce depot contient une API RAG Python basee sur FastAPI, LangChain, Ollama et ChromaDB. Le systeme ingere des documents HTML/textes depuis `data/`, cree une base vectorielle Chroma, puis expose un endpoint `/ask` qui retourne une reponse, des sources et des metadonnees.

Les fichiers les plus importants sont :

- `server.py` : API FastAPI, initialisation globale de `SimpleRAG`, endpoints `/health` et `/ask`.
- `rag_pipeline.py` : pipeline principal de retrieval, reranking, expansion de requete, detection d'hallucination et generation Ollama.
- `config.py` : configuration centrale des modeles, du retrieval, des embeddings et du chemin Chroma.
- `ingest_html_adaptive.py` : ingestion HTML avec chunking adaptatif et reconstruction de sections.
- `device_config.py` : selection CPU/GPU via `RAG_USE_GPU`.
- `reranker.py`, `query_expander.py`, `hallucination_detector.py`, `cache_manager.py`, `prompt_builder.py` : modules d'optimisation du pipeline.
- `test_*.py` et `test_*.sh` : tests manuels, diagnostics et scripts de validation.

## Contexte D'execution

Le projet est developpe localement dans `c:\coding\rag`, mais plusieurs scripts sont ecrits pour la production Linux dans `/home/rag`.

Attention particuliere :

- `config.py` definit `CHROMA_DB_PATH = "/home/rag/chroma_db"`.
- `ingest_html_adaptive.py`, `check_chroma.py`, `reingest_adaptive.sh` et plusieurs scripts de debug utilisent aussi `/home/rag/chroma_db`.
- Le depot contient un dossier local `chroma_db/`, mais le code actif peut ne pas l'utiliser tant que les chemins absolus ne sont pas ajustes.
- `.gitignore` ignore `data/*`; les donnees source peuvent etre sensibles ou absentes du depot.

Avant de modifier un chemin de donnees ou de base vectorielle, verifier si l'objectif est local Windows, serveur Linux, ou les deux.

## Dependances

Il n'y a pas de `requirements.txt` dans l'etat actuel. Les dependances visibles dans le code et la documentation incluent :

- `fastapi`
- `uvicorn`
- `requests`
- `beautifulsoup4`
- `langchain`
- `langchain-community`
- `langchain-chroma`
- `langchain-ollama`
- `langchain-text-splitters`
- `chromadb`
- `pydantic`
- `torch` lorsque les modules GPU/reranking/detection l'utilisent
- Ollama avec au minimum `nomic-embed-text` et le modele LLM configure

Commandes d'installation indicatives :

```bash
pip install langchain langchain-community langchain-chroma langchain-ollama chromadb fastapi uvicorn requests beautifulsoup4 python-dotenv
ollama pull nomic-embed-text
ollama pull qwen2.5:14b
```

Adapter le modele Ollama a `config.py` ou aux variables d'environnement.

## Variables Et Configuration

Variables utiles :

- `RAG_MODEL` : surcharge le modele utilise par `server.py` au lancement.
- `RAG_MAX_CONTEXT` : surcharge la taille max du contexte dans `server.py`.
- `RAG_USE_GPU` : `auto` par defaut, `0`/`false`/`cpu` pour forcer CPU, `1`/`true`/`gpu` pour demander CUDA.

Configuration centrale :

- Changer le modele par defaut dans `config.py` via `ACTIVE_MODEL`.
- Ajuster le retrieval dans `RAG_CONFIG`.
- Ajuster le modele d'embeddings via `EMBEDDING_MODEL`.

Note : `server.py` force actuellement `CUDA_VISIBLE_DEVICES = ""`, ce qui desactive CUDA pour le processus API, meme si `device_config.py` sait detecter le GPU. Ne pas modifier ce comportement sans verifier la cible materielle et les modules dependants.

## Commandes Courantes

Demarrer l'API localement :

```bash
python server.py
```

Demarrer via Uvicorn :

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

Tester la sante de l'API :

```bash
curl http://localhost:8000/health
```

Interroger l'API :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quelles sont les regles des conges ?"}'
```

Lancer un test API direct :

```bash
python test_api_direct.py
```

Lancer la suite de questions de test :

```bash
python test_rag.py
```

Inspecter la base Chroma :

```bash
python check_chroma.py
python diagnostic_db.py
```

Reingestion adaptative sur serveur Linux :

```bash
bash reingest_adaptive.sh
```

Cette commande remplace la base Chroma cible apres confirmation interactive. Ne jamais la lancer pour "voir" sans comprendre le chemin cible.

## Architecture RAG

Flux principal :

1. `server.py` cree une instance globale `SimpleRAG`.
2. Au startup FastAPI, `rag.setup_chain()` initialise les embeddings Ollama, Chroma, le retriever et le LLM.
3. `/ask` appelle `rag.ask(question, return_sources=True, dynamic_k=...)`.
4. `rag_pipeline.py` recupere des documents via Chroma et, selon la configuration, BM25 hybride.
5. Les modules d'optimisation peuvent expanser la requete, reranker, filtrer et detecter les hallucinations.
6. La reponse finale est renvoyee avec `sources` et `metadata`.

Le retriever hybride privilegie la recherche vectorielle et active BM25 si les resultats sont insuffisants ou si des mots-cles metier sont detectes.

## Ingestion Et Donnees

`ingest_html_adaptive.py` :

- lit les fichiers depuis `data/`;
- extrait le contenu HTML avec BeautifulSoup;
- detecte les sections marquees avec `+-`;
- cree des chunks adaptes a la taille et au type du document;
- ecrit dans Chroma avec `OllamaEmbeddings(model="nomic-embed-text")`.

Garde-fous :

- Ne pas commiter de donnees metier issues de `data/`.
- Ne pas supprimer `chroma_db/` ou `/home/rag/chroma_db` sans sauvegarde explicite.
- Les scripts `.sh` sont majoritairement prevus pour Linux et peuvent contenir des chemins absolus `/home/rag`.
- Si vous adaptez le projet pour Windows, centraliser les chemins plutot que dupliquer des constantes.

## Tests Et Verification

Il n'y a pas de framework de test unique. Choisir la verification selon le changement :

- Changement API : `python server.py`, puis `/health`, puis `python test_api_direct.py`.
- Changement retrieval ou prompt : `python test_rag.py` avec `test_questions.json`.
- Changement Chroma/ingestion : `python ingest_html_adaptive.py` sur une copie controlee, puis `python check_chroma.py` ou `python test_chroma_load.py`.
- Changement GPU/CPU : `python device_config.py` ou import des modules concernes avec `RAG_USE_GPU` explicite.

Les tests RAG dependent d'Ollama, des modeles installes, de la base Chroma et des donnees locales. Un echec de connexion a Ollama ou a l'API n'est pas forcement une regression de code.

## Style Et Conventions

- Le code et les commentaires sont majoritairement en francais.
- Conserver les interfaces publiques existantes (`SimpleRAG.ask`, endpoint `/ask`, schema des sources) sauf demande explicite.
- Preferer les modifications ciblees : ce projet contient plusieurs variantes historiques (`rag_pipeline_old.py`, `rag_pipeline_backup.py`, `rag_pipeline_v2.py`). Le pipeline actif est `rag_pipeline.py`.
- Eviter les refactorings larges pendant un correctif de retrieval ou de prompt.
- Garder les sorties utilisateur utiles pour le diagnostic, mais eviter d'ajouter du bruit dans les chemins critiques.
- Si vous ajoutez une dependance, documenter la commande d'installation ou creer un `requirements.txt` dans le meme changement.

## Pieges Connus

- Plusieurs fichiers affichent des caracteres mal encodes dans les messages (`ðŸ...`, `Ã©`, etc.). Ne pas melanger correction d'encodage et changement fonctionnel sauf objectif explicite.
- Les chemins Chroma sont hardcodes a plusieurs endroits.
- `server.py` force le CPU via `CUDA_VISIBLE_DEVICES`.
- `find_ensemble.py`, `OPTIMISATION_RETRIEVAL.md`, `test_init.py` et `test_optimizations.py` sont vides.
- Des fichiers `*:Zone.Identifier` sont presents, probablement issus de Windows. Ne pas les utiliser comme sources fonctionnelles.
- Les scripts de redemarrage utilisent `pkill` et `nohup`; ils sont destines a Linux, pas a PowerShell.

## Avant De Rendre La Main

Pour une modification de code, fournir :

- les fichiers modifies;
- la commande de verification executee;
- les limites de verification si Ollama, Chroma ou les donnees ne sont pas disponibles;
- toute migration ou reingestion necessaire.

Pour une modification RAG qualitative, inclure au moins une question de test representative et mentionner les sources/metadonnees observees si l'API a ete lancee.
