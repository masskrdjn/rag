# Comparaison des Versions d'Ingestion HTML

## Vue d'ensemble

Vous disposez maintenant de **2 versions** du système d'ingestion :

1. **Version Standard** (`ingest_html.py`)
2. **Version Optimisée** (`ingest_html_optimized.py`) ⭐ RECOMMANDÉE

---

## Tableau Comparatif

| Caractéristique | Version Standard | Version Optimisée ⭐ |
|----------------|------------------|---------------------|
| **Chunk Size** | 1000 caractères | 800 caractères |
| **Chunk Overlap** | 200 caractères (20%) | 100 caractères (12.5%) |
| **Séparateurs** | Par défaut | Personnalisés (listes, titres) |
| **Extraction d'images** | ❌ Non | ✅ Oui |
| **Optimisé pour WordPress** | ❌ Non | ✅ Oui |
| **Dépendances** | `unstructured` | `unstructured` + `beautifulsoup4` |

---

## Différences Détaillées

### 1. Paramètres de Chunking

#### Version Standard
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
```
- Chunks plus grands = moins de précision
- Overlap 20% = plus de redondance

#### Version Optimisée ⭐
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)
```
- Chunks optimaux pour des procédures
- Overlap 12.5% = bon équilibre performance/précision

---

### 2. Séparateurs

#### Version Standard
Utilise les séparateurs par défaut de LangChain :
- `\n\n` (paragraphes)
- `\n` (lignes)
- ` ` (espaces)

#### Version Optimisée ⭐
Séparateurs adaptés au contenu structuré WordPress :
```python
separators=[
    "\n\n",          # Paragraphes
    "\n- ",          # Listes à puces
    "\n* ",          # Listes alternatives
    "\n# ",          # Titres markdown H1
    "\n## ",         # Titres markdown H2
    "\n### ",        # Titres markdown H3
    "\n",            # Lignes simples
    " ",             # Mots
    "",              # Caractères
]
```

**Avantage** : Les chunks ne coupent pas au milieu d'une liste ou d'un titre !

---

### 3. Gestion des Images 🖼️

#### Version Standard
```python
# Utilise UnstructuredHTMLLoader
loader = DirectoryLoader(
    DATA_PATH, 
    glob="**/*.html",
    loader_cls=UnstructuredHTMLLoader
)
```
**Résultat** : Les balises `<img>` sont **supprimées**. Vous perdez les références aux images.

#### Version Optimisée ⭐
```python
def extract_images_from_html(html_content, source_file):
    """Extrait les URLs d'images et les ajoute au texte"""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    # Extraire les images
    images = soup.find_all('img')
    if images:
        text += "\n\n=== IMAGES RÉFÉRENCÉES ===\n"
        for idx, img in enumerate(images, 1):
            src = img.get('src', '')
            alt = img.get('alt', 'Sans description')
            text += f"\nImage {idx}: {alt}\nURL: {src}\n"
    
    return text
```

**Résultat** : Les URLs d'images sont **extraites et indexées**. Exemple de texte indexé :

```
Procédure de configuration WordPress...

=== IMAGES RÉFÉRENCÉES ===

Image 1: Capture écran du tableau de bord
URL: https://example.com/wp-content/uploads/dashboard.png

Image 2: Menu de configuration
URL: https://example.com/wp-content/uploads/menu-config.png
```

---

## Cas d'Usage : Quelle Version Choisir ?

### Utilisez la **Version Standard** si :
- Vous avez des documents HTML **simples** sans images importantes
- Vous voulez une installation **minimale** (sans BeautifulSoup)
- Vos documents ne contiennent **pas de listes** ou structure complexe

### Utilisez la **Version Optimisée** ⭐ si :
- Vos documents contiennent des **captures d'écran WordPress**
- Vous avez des **procédures avec listes** étape par étape
- Vous voulez la **meilleure précision** de recherche
- Vos documents ont une **structure markdown/HTML riche**

---

## Impact sur les Requêtes

### Exemple : "Où se trouve l'image du tableau de bord ?"

#### Version Standard
```
❌ Réponse : "Je ne sais pas, les images ne sont pas référencées."
```

#### Version Optimisée ⭐
```
✅ Réponse : "L'image du tableau de bord se trouve à l'URL :
https://example.com/wp-content/uploads/dashboard.png
Elle est décrite comme 'Capture écran du tableau de bord'."
```

---

## Performance & Taille de la Base

### Taille des Chunks

**100 fichiers HTML de 5000 caractères chacun :**

| Version | Chunks créés | Taille DB estimée |
|---------|-------------|-------------------|
| Standard (1000/200) | ~625 chunks | Plus grande |
| Optimisée (800/100) | ~688 chunks | Moyenne |

**Conclusion** : Légèrement plus de chunks avec la version optimisée, mais meilleure qualité de récupération.

---

## Migration entre Versions

### De Standard → Optimisée

```bash
# 1. Supprimer l'ancienne base
rm -rf /home/rag/chroma_db/

# 2. Installer BeautifulSoup (si pas déjà fait)
pip3 install beautifulsoup4

# 3. Ré-ingérer les données
cd /home/rag && python3 ingest_html_adaptive.py

# 4. Redémarrer le serveur
bash /home/rag/restart_server.sh
```

---

## Recommandation Finale

> [!IMPORTANT]
> **Pour vos procédures WordPress avec captures d'écran, utilisez ABSOLUMENT la version optimisée.**
> 
> Les conseils que vous avez reçus sont pertinents et déjà intégrés dans `ingest_html_optimized.py`.

### Commande à utiliser

```bash
# Rendre exécutable
chmod +x /home/rag/ingest_html_optimized_workflow.sh

# Lancer l'ingestion optimisée
sudo bash /home/rag/ingest_html_optimized_workflow.sh /chemin/vers/vos/html
```

---

## Fichiers Disponibles

### Scripts Python
- `ingest.py` - Version originale (texte seulement)
- `ingest_html.py` - Version HTML standard
- `ingest_html_optimized.py` ⭐ - Version HTML optimisée (RECOMMANDÉE)

### Scripts Bash
- `ingest_html_workflow.sh` - Workflow standard
- `ingest_html_optimized_workflow.sh` ⭐ - Workflow optimisé (RECOMMANDÉE)

### Documentation
- `INGESTION_HTML.md` - Guide d'utilisation
- `COMPARAISON_VERSIONS.md` - Ce fichier

---

## Support & Dépannage

### Erreur : "No module named 'bs4'"
**Solution** : Installer BeautifulSoup4
```bash
pip3 install beautifulsoup4
```

### Les images ne sont pas indexées
**Solution** : Vérifiez que vous utilisez bien `ingest_html_adaptive.py`

### Vérifier quelle version est utilisée
```bash
# Regarder le nom du script dans les logs
cat /home/rag/server.log
```
