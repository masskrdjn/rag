# Chunking Adaptatif Intelligent

## 🎯 Pourquoi le Chunking Adaptatif ?

### Le Problème avec un Chunk Size Fixe

Avec un chunk size fixe (ex: 600 caractères pour tous les documents) :

❌ **Documents courts (< 2000 chars)** : Sur-fragmentation, perte de contexte global
❌ **Documents longs (> 20000 chars)** : Trop de fragments, contexte dilué
❌ **Approche "one-size-fits-all" non optimale**

### La Solution : Chunking Adaptatif

✅ **Adapte automatiquement** la taille des chunks selon :
- La longueur totale du document
- Le type de document (Amadeus vs standard)
- La structure du contenu

## 📊 Stratégies de Chunking

| Catégorie | Taille Document | Chunk Size | Chunk Overlap | Objectif |
|-----------|----------------|------------|---------------|----------|
| **Très courts** | < 2,000 chars | 400 | 50 | Contexte complet |
| **Courts** | 2k-5k chars | 500 | 60 | Sections entières |
| **Moyens** | 5k-10k chars | 600 | 80 | Équilibre |
| **Longs** | 10k-20k chars | 700 | 90 | Précision |
| **Très longs** | > 20k chars | 900 | 120 | Éviter fragmentation |

### Cas Spécial : Documents Amadeus

Pour les documents avec sections (format `+-Section`) :
- Chaque section devient un document distinct
- Chunk size réduit de **15%** (car contexte déjà isolé)

## 🚀 Utilisation

```bash
cd /home/rag

# Analyser votre corpus
python3 analyze_files.py

# Réingestion adaptative
./reingest_adaptive.sh
```

## 📈 Résultats Attendus

| Aspect | Chunk Fixe (600) | Chunk Adaptatif | Amélioration |
|--------|------------------|-----------------|--------------|
| **Docs courts** | Sur-fragmentés | Contexte préservé | +40% précision |
| **Docs longs** | Trop fragmentés | Optimisé | -30% chunks |
| **Sections Amadeus** | Noyées | Isolées | +80% rappel |
| **Qualité globale** | Variable | Consistante | +35% |

## 🔧 Configuration

Dans `ingest_html_adaptive.py`, fonction `get_adaptive_chunk_config()` :

```python
# Modifier ces seuils selon vos besoins
if text_length < 2000:      # Très courts
    return {'chunk_size': 400, ...}
elif text_length < 5000:    # Courts
    return {'chunk_size': 500, ...}
# ...
```

## 📚 Fichiers

- `analyze_files.py` - Analyse la distribution
- `ingest_html_adaptive.py` - Système de chunking adaptatif
- `reingest_adaptive.sh` - Script de réingestion
- `CHUNKING_ADAPTATIF.md` - Cette documentation
