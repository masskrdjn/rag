import os
import re
import shutil
import hashlib
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DB_PATH, DATA_PATH, EMBEDDING_MODEL
from embeddings_factory import get_embeddings

# Alias historiques pour rétrocompatibilité avec les fonctions ci-dessous
CHROMA_PATH = CHROMA_DB_PATH


def make_stable_id(*parts) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def clear_chroma_db():
    """Supprime la base de données existante pour éviter la corruption."""
    if os.path.exists(CHROMA_PATH):
        print(f"Cleaning up existing ChromaDB at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)
        print("✓ Cleanup complete")

def get_adaptive_chunk_config(text_length, has_sections=False):
    """
    Retourne une configuration de chunking adaptée à la taille du document.
    
    Args:
        text_length: Nombre de caractères du document
        has_sections: True si le document a des sections Amadeus (+-Section)
    
    Returns:
        dict: {chunk_size, chunk_overlap, description}
    """
    configs = {
        'no_chunk': {
            'chunk_size': 0,  # 0 signale "pas de chunking"
            'chunk_overlap': 0,
            'description': 'Document entier - contexte maximal'
        },
        'very_short': {
            'chunk_size': 600,
            'chunk_overlap': 100,
            'description': 'Très court - contexte étendu'
        },
        'short': {
            'chunk_size': 800,
            'chunk_overlap': 150,
            'description': 'Court - sections larges'
        },
        'medium': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'description': 'Moyen - bon équilibre'
        },
        'long': {
            'chunk_size': 1200,
            'chunk_overlap': 250,
            'description': 'Long - précision élevée'
        },
        'very_long': {
            'chunk_size': 1500,
            'chunk_overlap': 300,
            'description': 'Très long - éviter fragmentation'
        }
    }
    
    # Pour les documents avec sections (Amadeus), on réduit légèrement
    # car chaque section est déjà un document distinct
    if has_sections:
        for config in configs.values():
            config['chunk_size'] = int(config['chunk_size'] * 0.85)
            config['chunk_overlap'] = int(config['chunk_overlap'] * 0.85)
    
    # Sélection basée sur la longueur
    if text_length < 1500:
        return configs['no_chunk']
    elif text_length < 3000:
        return configs['very_short']
    elif text_length < 6000:
        return configs['short']
    elif text_length < 12000:
        return configs['medium']
    elif text_length < 25000:
        return configs['long']
    else:
        return configs['very_long']

def extract_process_sections(text, source_file):
    """
    Extrait les sections de TOUS les documents avec markers +-
    avec chunking hiérarchique : document complet + sections individuelles.
    
    Cette approche permet:
    - Questions générales → document de synthèse trouvé
    - Questions spécifiques → section précise trouvée
    - Reconstruction de l'ordre logique grâce aux métadonnées
    
    Args:
        text: Texte du document
        source_file: Chemin du fichier source
    
    Returns:
        list: Liste de sections avec contenu enrichi, ou None si aucune section
    """
    sections = []
    
    # Pattern pour détecter les sections (format +- Titre)
    section_pattern = r'\+-([^\n]+)'
    
    # Trouver toutes les sections
    matches = list(re.finditer(section_pattern, text))
    
    if not matches:
        return None
    
    # Détecter le type de document
    is_amadeus = "Formats utiles Amadeus" in text or "Formats et TUTO utiles" in text
    is_process = any(keyword in text for keyword in ["Process", "Étape", "Procédure", "Etape"])
    
    doc_type = "amadeus" if is_amadeus else ("process" if is_process else "structured")
    
    # Extraire le titre du document
    title_match = re.search(r'Titre du document : ([^\n]+)', text)
    doc_title = title_match.group(1) if title_match else "Document"
    
    # Nombre total de sections pour le contexte
    total_sections = len(matches)
    
    print(f"  ✓ Document {doc_type} détecté avec {total_sections} sections")
    
    # === DOCUMENT DE SYNTHÈSE (index/sommaire) ===
    # Sommaire = index pour les questions générales (« comment utiliser X »).
    # On garde juste le titre du doc + intro + liste des titres de sections.
    # On N'inclut PLUS le contenu des sections courtes (gérées comme sections
    # individuelles), pour éviter qu'un long sommaire concurrence les chunks
    # précis lors du retrieval.
    toc_lines = [f"Document: {doc_title}", "", "Sommaire:"]

    intro_end = matches[0].start()
    intro_text = text[:intro_end].strip()
    intro_text = re.sub(r'Titre du document : [^\n]+\n?', '', intro_text).strip()
    if intro_text and len(intro_text) > 50:
        toc_lines.append("")
        toc_lines.append("Introduction :")
        toc_lines.append(intro_text[:400])

    toc_lines.append("")
    toc_lines.append("Sections :")
    for i, match in enumerate(matches, 1):
        section_title = match.group(1).strip()
        toc_lines.append(f"  {i}. {section_title}")

    sections.append({
        'content': "\n".join(toc_lines),
        'title': 'Sommaire',
        'section_num': 0,  # 0 = sommaire, toujours en premier
        'total_sections': total_sections,
        'doc_type': doc_type,
        'is_summary': True,
        # Pas de voisins pour le sommaire — il couvre tout le doc.
        'doc_title': doc_title,
        'prev_section_title': '',
        'next_section_title': '',
    })

    # === SECTIONS INDIVIDUELLES ===
    # Le contenu indexé contient seulement le titre de section + le contenu.
    # Le contexte hiérarchique (doc parent, voisins) passe en métadonnée pour
    # ne PAS polluer l'embedding avec du boilerplate quasi-identique entre les
    # sections d'un même document.
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_pos = match.start()

        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        section_content = text[start_pos:end_pos].strip()
        # Nettoyage du marker `+-Titre` lui-même (déjà capturé dans le titre).
        section_content = re.sub(r'^\+-[^\n]*\n?', '', section_content).strip()

        prev_title = matches[i - 1].group(1).strip() if i > 0 else ''
        next_title = matches[i + 1].group(1).strip() if i < len(matches) - 1 else ''

        # Contenu indexé : titre + texte. Pas de header verbeux.
        indexed_content = f"{section_title}\n\n{section_content}"

        sections.append({
            'content': indexed_content,
            'title': section_title,
            'section_num': i + 1,
            'total_sections': total_sections,
            'doc_type': doc_type,
            'is_summary': False,
            'doc_title': doc_title,
            'prev_section_title': prev_title,
            'next_section_title': next_title,
        })

    return sections

def extract_images_from_html(html_content, source_file):
    """
    Extrait les URLs d'images du HTML et les ajoute au texte
    pour qu'elles soient indexées dans la base vectorielle.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extraire le titre
    title = soup.title.string if soup.title else ""
    
    # Supprimer les scripts et styles
    for script in soup(["script", "style"]):
        script.decompose()
    
    text = soup.get_text(separator='\n')
    
    # Enrichir le contexte pour les documents courts ou de redirection
    if title:
        text = f"Titre du document : {title}\n\n{text}"
        
    if "Whaller" in text and "Consulter" in text:
        text += "\n\nNOTE IMPORTANTE : Pour les détails complets et les règles officielles, il est impératif de consulter la sphère Whaller indiquée ci-dessus."
    
    # Extraire les images
    images = soup.find_all('img')
    
    if images:
        text += "\n\n=== IMAGES RÉFÉRENCÉES ===\n"
        for idx, img in enumerate(images, 1):
            src = img.get('src', '')
            alt = img.get('alt', 'Sans description')
            if src:
                text += f"\nImage {idx}: {alt}\nURL: {src}\n"
    
    return text

def extract_html_metadata(html_content):
    """
    Extrait les métadonnées du HTML (post-id, catégorie, date, etc.)
    
    Args:
        html_content: Contenu HTML brut
    
    Returns:
        dict: Métadonnées extraites
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    metadata = {}
    
    # Mapping des meta tags vers les clés de métadonnées
    meta_tags = {
        'post-id': 'post_id',
        'date': 'created_date',
        'modified': 'modified_date', 
        'categories': 'category',
        'tags': 'tags',
        'source-url': 'source_url',
        'slug': 'slug'
    }
    
    for meta_name, key in meta_tags.items():
        meta = soup.find('meta', attrs={'name': meta_name})
        if meta and meta.get('content'):
            metadata[key] = meta.get('content', '').strip()
    
    return metadata


_DATE_REGEX = re.compile(
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'        # JJ/MM/AAAA
    r'|\b\d{4}-\d{2}-\d{2}\b'                    # ISO
    r'|\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|aoû?t|septembre|octobre|novembre|décembre)\b',
    flags=re.IGNORECASE,
)


def _compute_freshness(modified_date: str) -> float:
    """
    Score [0, 1] décroissant avec l'âge du document. Renvoie 0.5 si la date
    n'est pas parseable, pour ne pas pénaliser injustement.
    """
    if not modified_date:
        return 0.5
    try:
        # Format ISO attendu : "2017-09-27T16:00:33"
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(modified_date.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).days
        if age_days < 365:
            return 1.0
        if age_days < 365 * 3:
            return 0.8
        if age_days < 365 * 5:
            return 0.6
        if age_days < 365 * 8:
            return 0.4
        return 0.3
    except (ValueError, TypeError):
        return 0.5


def enrich_document_metadata(section: dict, html_metadata: dict) -> dict:
    """Enrichir les métadonnées pour meilleure pertinence."""

    enhanced = html_metadata.copy()

    content = section['content']
    content_lower = content.lower()

    # 1. Signaux de fiabilité/importance — désormais alignés sur des proxys
    # robustes plutôt que des heuristiques boolean-presque-toujours-True.
    reliability_signals = {
        'has_official_link': 1.0 if 'whaller' in content_lower else 0.5,
        'is_procedure': 1.0 if any(
            w in content_lower for w in ('étape', 'procédure', 'process', 'comment')
        ) else 0.5,
        'has_date_info': 1.0 if _DATE_REGEX.search(content) else 0.5,
        'document_freshness': _compute_freshness(html_metadata.get('modified_date', '')),
    }

    enhanced['reliability_score'] = sum(reliability_signals.values()) / len(reliability_signals)
    enhanced['reliability_signals'] = str(reliability_signals)

    # 2. Catégories automatiques (classification grossière à la lecture)
    categories = []
    if 'amadeus' in content_lower or 'gds' in content_lower or 'format' in content_lower:
        categories.append('amadeus_formats')
    if 'congé' in content_lower or 'absence' in content_lower:
        categories.append('absences')
    if 'émission' in content_lower or 'dossier' in content_lower:
        categories.append('gds_operations')

    enhanced['auto_categories'] = str(categories)
    enhanced['category_string'] = ' '.join(categories)

    # 3. Densité informationnelle (utilisée par le reranker)
    words = content.split()
    unique_words = len(set(words))
    enhanced['information_density'] = unique_words / max(len(words), 1)

    # 4. Taille normalisée
    enhanced['chunk_size_kb'] = len(content) / 1024

    # 5. Contexte hiérarchique de la section, en MÉTADONNÉE et non plus dans
    # le page_content (évite de polluer l'embedding avec du boilerplate).
    for key in ('doc_title', 'prev_section_title', 'next_section_title'):
        if key in section:
            enhanced[key] = section[key]

    return enhanced


def load_html_documents_adaptive(data_path):
    """
    Charge tous les fichiers HTML avec chunking adaptatif.
    - Documents Amadeus : séparés par sections
    - Autres documents : chunking adapté à leur taille
    """
    documents = []
    data_dir = Path(data_path)
    
    # Chercher récursivement tous les fichiers HTML
    html_files = list(data_dir.rglob("*.html")) + list(data_dir.rglob("*.htm"))
    
    print(f"Found {len(html_files)} HTML file(s)")
    print("="*80)
    
    stats = {
        'amadeus_sections': 0,
        'no_chunk': 0,
        'very_short': 0,
        'short': 0,
        'medium': 0,
        'long': 0,
        'very_long': 0
    }
    
    for html_file in html_files:
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extraire les métadonnées HTML
            html_metadata = extract_html_metadata(html_content)
            
            # Extraire texte + images
            text = extract_images_from_html(html_content, str(html_file))
            text_length = len(text)
            
            # Tenter d'extraire les sections avec markers +-
            sections = extract_process_sections(text, str(html_file))
            
            if sections:
                # Créer un document par section (incluant le sommaire)
                for section in sections:
                    section_length = len(section['content'])
                    chunk_config = get_adaptive_chunk_config(section_length, has_sections=True)
                    parent_id = make_stable_id(html_file.name, html_metadata.get("post_id"), "parent")
                    child_id = make_stable_id(
                        html_file.name,
                        html_metadata.get("post_id"),
                        section.get('section_num'),
                        section.get('title'),
                    )
                    
                    # Enrichir les métadonnées
                    enriched_meta = enrich_document_metadata(section, html_metadata)
                    
                    # Type de document plus précis
                    if section.get('is_summary'):
                        doc_type_str = f"html_{section['doc_type']}_summary"
                    else:
                        doc_type_str = f"html_{section['doc_type']}_section"
                    
                    doc = Document(
                        page_content=section['content'],
                        metadata={
                            "source": str(html_file),
                            "filename": html_file.name,
                            "type": doc_type_str,
                            "section_title": section['title'],
                            "section_num": section['section_num'],
                            "total_sections": section.get('total_sections', 0),
                            "parent_id": parent_id,
                            "child_id": child_id,
                            "is_summary": section.get('is_summary', False),
                            "doc_type": section['doc_type'],
                            "chunk_size": chunk_config['chunk_size'],
                            "chunk_overlap": chunk_config['chunk_overlap'],
                            "text_length": section_length,
                            **enriched_meta  # Ajouter les métadonnées enrichies
                        }
                    )
                    documents.append(doc)
                    stats['amadeus_sections'] += 1
                
                # +1 pour le sommaire
                print(f"✓ {html_file.name}: {len(sections)} chunks ({len(sections)-1} sections + 1 sommaire) [{section['doc_type']}]")
            else:
                # Traitement normal avec chunking adaptatif
                chunk_config = get_adaptive_chunk_config(text_length, has_sections=False)
                
                # Enrichir les métadonnées (simulé pour doc entier)
                dummy_section = {'content': text}
                enriched_meta = enrich_document_metadata(dummy_section, html_metadata)
                
                # Catégoriser pour les stats
                if text_length < 2000:
                    category = 'very_short'
                elif text_length < 5000:
                    category = 'short'
                elif text_length < 10000:
                    category = 'medium'
                elif text_length < 20000:
                    category = 'long'
                else:
                    category = 'very_long'
                
                stats[category] += 1
                
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": str(html_file),
                        "filename": html_file.name,
                        "type": "html",
                        "parent_id": make_stable_id(html_file.name, html_metadata.get("post_id"), "parent"),
                        "child_id": make_stable_id(html_file.name, html_metadata.get("post_id"), "full"),
                        "chunk_size": chunk_config['chunk_size'],
                        "chunk_overlap": chunk_config['chunk_overlap'],
                        "chunk_strategy": chunk_config['description'],
                        "text_length": text_length,
                        **enriched_meta  # Ajouter les métadonnées enrichies
                    }
                )
                documents.append(doc)
                
                print(f"✓ {html_file.name}: {text_length:,} chars → {chunk_config['description']} (chunk={chunk_config['chunk_size']})")
            
        except Exception as e:
            print(f"✗ Error loading {html_file.name}: {e}")
    
    print("="*80)
    print(f"\n📊 STATISTIQUES DE CHARGEMENT:")
    print(f"   • Sections Amadeus:     {stats['amadeus_sections']:3d}")
    print(f"   • Documents entiers:     {stats['no_chunk']:3d} (pas de chunking)")
    print(f"   • Documents très courts: {stats['very_short']:3d} (chunk=600)")
    print(f"   • Documents courts:      {stats['short']:3d} (chunk=800)")
    print(f"   • Documents moyens:      {stats['medium']:3d} (chunk=1000)")
    print(f"   • Documents longs:       {stats['long']:3d} (chunk=1200)")
    print(f"   • Documents très longs:  {stats['very_long']:3d} (chunk=1500)")
    print(f"   • TOTAL:                 {len(documents):3d} documents\n")
    
    return documents

def chunk_documents_adaptive(documents):
    """
    Applique un chunking adaptatif basé sur les métadonnées de chaque document.
    """
    all_chunks = []
    
    # Grouper les documents par configuration de chunk
    doc_groups = {}
    for doc in documents:
        chunk_size = doc.metadata.get('chunk_size', 600)
        chunk_overlap = doc.metadata.get('chunk_overlap', 80)
        key = (chunk_size, chunk_overlap)
        
        if key not in doc_groups:
            doc_groups[key] = []
        doc_groups[key].append(doc)
    
    print(f"Chunking avec {len(doc_groups)} configurations différentes...")
    
    # Séparateurs optimisés
    separators = [
        "\n\n",            # Paragraphes
        "\n+-",            # Sections Amadeus
        "\n- ",            # Listes à puces
        "\n* ",            # Listes alternatives
        "\n# ",            # Titres markdown H1
        "\n## ",           # Titres markdown H2
        "\n### ",          # Titres markdown H3
    ]
    
    # Appliquer le chunking par groupe
    for (chunk_size, chunk_overlap), group_docs in doc_groups.items():
        # Cas spécial : pas de chunking (documents entiers)
        if chunk_size == 0:
            print(f"  • Documents entiers: {len(group_docs)} docs → {len(group_docs)} chunks")
            all_chunks.extend(group_docs)
            continue
            
        print(f"  • Chunk size {chunk_size}/{chunk_overlap}: {len(group_docs)} docs", end="")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            add_start_index=True,
        )
        
        chunks = text_splitter.split_documents(group_docs)
        for index, chunk in enumerate(chunks):
            base_child_id = chunk.metadata.get("child_id") or make_stable_id(
                chunk.metadata.get("filename"),
                chunk.metadata.get("section_num"),
                chunk.metadata.get("start_index"),
            )
            chunk.metadata["child_id"] = make_stable_id(base_child_id, chunk.metadata.get("start_index"), index)
            chunk.metadata["parent_id"] = chunk.metadata.get("parent_id") or make_stable_id(
                chunk.metadata.get("filename"), "parent"
            )
        all_chunks.extend(chunks)
        
        print(f" → {len(chunks)} chunks")
    
    return all_chunks

def ingest_html_documents_adaptive():
    """
    Ingestion avec chunking adaptatif intelligent:
    - Détection automatique des documents Amadeus (sections)
    - Chunking adapté à la taille de chaque document
    - Optimisation pour documents courts vs longs
    """
    print(f"\n{'='*80}")
    print(f"INGESTION ADAPTATIVE - Chargement depuis {DATA_PATH}")
    print(f"{'='*80}\n")
    
    # Nettoyer la base existante
    clear_chroma_db()
    
    # Charger les documents avec métadonnées de chunking
    documents = load_html_documents_adaptive(DATA_PATH)
    
    if len(documents) == 0:
        print("⚠ No HTML documents found. Check your data path.")
        return
    
    # Calculer la taille totale du texte
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"Total characters to process: {total_chars:,}")
    
    # Appliquer le chunking adaptatif
    print(f"\n{'='*80}")
    print("CHUNKING ADAPTATIF")
    print(f"{'='*80}")
    chunks = chunk_documents_adaptive(documents)
    
    avg_chunk_size = total_chars // len(chunks) if chunks else 0
    print(f"\n✓ Total chunks créés: {len(chunks)}")
    print(f"✓ Taille moyenne: {avg_chunk_size} caractères")
    
    # Afficher un échantillon d'une section Amadeus
    amadeus_chunk = next((c for c in chunks if 'repas' in c.page_content.lower() or 'srr' in c.page_content.lower()), None)
    if amadeus_chunk:
        print(f"\n{'='*80}")
        print("ÉCHANTILLON: Section Repas Spéciaux")
        print(f"{'='*80}")
        print(f"Source: {amadeus_chunk.metadata.get('filename', 'N/A')}")
        if 'section_title' in amadeus_chunk.metadata:
            print(f"Section: {amadeus_chunk.metadata['section_title']}")
        print(f"Chunk config: {amadeus_chunk.metadata.get('chunk_size', 'N/A')}/{amadeus_chunk.metadata.get('chunk_overlap', 'N/A')}")
        print(f"\nContenu:")
        print(amadeus_chunk.page_content[:400])
        print(f"{'='*80}")
    
    # Sauvegarder dans ChromaDB
    print(f"\n{'='*80}")
    print("SAUVEGARDE DANS CHROMADB")
    print(f"{'='*80}")
    embeddings = get_embeddings(EMBEDDING_MODEL)
    
    # Créer/mettre à jour le vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    # Force persistence if available
    try:
        if hasattr(vectorstore, 'persist'):
            vectorstore.persist()
            print("✓ Base persistée sur disque")
    except Exception as e:
        print(f"⚠ Persistence explicite non disponible: {e}")
    
    print(f"\n{'='*80}")
    print("✅ INGESTION ADAPTATIVE TERMINÉE!")
    print(f"{'='*80}")
    print(f"\n📊 RÉSUMÉ:")
    print(f"  • Documents sources:     {len(documents)}")
    print(f"  • Chunks créés:          {len(chunks)}")
    print(f"  • Taille moyenne chunk:  {avg_chunk_size} chars")
    print(f"  • Modèle d'embedding:    {EMBEDDING_MODEL}")
    print(f"  • Base de données:       {CHROMA_PATH}")
    print(f"\n💡 AVANTAGES:")
    print(f"  ✓ Chunking adapté à chaque type de document")
    print(f"  ✓ Documents courts: contexte maximal préservé")
    print(f"  ✓ Documents longs: précision optimale")
    print(f"  ✓ Sections Amadeus: séparées et contextualisées")
    print(f"  ✓ Images indexées et recherchables")
    print(f"\n⚠️  N'oubliez pas de redémarrer votre serveur RAG!")

if __name__ == "__main__":
    ingest_html_documents_adaptive()
