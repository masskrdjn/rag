# Migration — repasse 2026-05-06

Notes de migration consécutives à la repasse d'optimisation du 2026-05-06
(chunking, retrieval, prompt, anti-hallucination, dette technique).

## Réingestion requise

Plusieurs changements modifient la représentation indexée des documents et
**imposent de relancer `python ingest_html_adaptive.py` pour produire une
base Chroma cohérente** (les anciens vecteurs ne sont pas compatibles) :

1. **Préfixage embeddings** (`embeddings_factory.py`) — pour
   `nomic-embed-text`, les passages sont désormais préfixés
   `search_document: ` à l'ingestion et les requêtes `search_query: ` au
   retrieval. C'est l'usage attendu par Nomic ; l'ancienne base ne contient
   pas ces préfixes.
2. **Header de section retiré du `page_content`** (`ingest_html_adaptive.py`)
   — le contexte hiérarchique (titre du document parent, sections voisines)
   ne pollue plus l'embedding. Il est ré-injecté côté LLM via les
   métadonnées (`doc_title`, `prev_section_title`, `next_section_title`,
   attribut `position` du bloc `<source>`).
3. **Sommaire allégé** — le chunk « sommaire » (section 0) ne réplique plus
   le contenu des sections courtes ; il ne contient que le titre du doc,
   l'introduction et la liste des titres de sections.
4. **`reliability_signals.document_freshness`** est désormais calculé à
   partir de la meta `modified` HTML, et `has_date_info` repose sur un
   regex date plutôt que sur la simple présence d'un chiffre.

### Procédure

```bash
python ingest_html_adaptive.py
python clear_cache.py            # le cache pré-repasse n'est plus comparable
# puis redémarrer l'API
```

## Autres changements de cette repasse

- **`hybrid_weights` par défaut : `[0.5, 0.5]`** (BM25/vector). À
  benchmarker contre `[0.4, 0.6]` ou `[0.6, 0.4]` avec
  `python test_rag.py` selon ton corpus.
- **`num_predict` uniformisé à 1024** pour tous les modèles (les procédures
  à 8-15 étapes étaient parfois tronquées à 512).
- **Prompt système réécrit** avec accents UTF-8, exemple few-shot intégré,
  format markdown imposé pour les questions procédurales.
- **Détecteur d'hallucinations** : factorisation du `tfidf.fit` (cache par
  source citée), filtrage des petits entiers communs dans les tokens
  sensibles, restauration du filtrage par patterns d'assertion sur
  `extract_claims`.
- **Cache** : la clé n'inclut plus `top_k` (le `top_k` réel est dynamique).
  Le cache existant sera silencieusement ignoré.
- **`server.py`** : `SimpleRAG` est instancié dans le `lifespan`, plus à
  l'import du module — facilite les tests et n'expose plus Ollama au
  premier import.
- **Code mort supprimé** : `_ask_legacy`, `_legacy_create_retriever`,
  `_create_hybrid_retriever_smart` (deux variantes), `_estimate_dynamic_topk_legacy`,
  `_format_docs_for_context_legacy`, `_build_generation_prompt_legacy`, et
  les définitions doublonnées de `_estimate_dynamic_topk`,
  `_format_docs_for_context` et `extract_claims`.
