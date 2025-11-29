import os
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import re

# Configuration
DATA_PATH = "/home/ragapp/rag-system/data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

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
        'very_short': {
            'chunk_size': 400,
            'chunk_overlap': 50,
            'description': 'Très court - priorité au contexte complet'
        },
        'short': {
            'chunk_size': 500,
            'chunk_overlap': 60,
            'description': 'Court - sections complètes'
        },
        'medium': {
            'chunk_size': 600,
            'chunk_overlap': 80,
            'description': 'Moyen - bon équilibre'
        },
        'long': {
            'chunk_size': 700,
            'chunk_overlap': 90,
            'description': 'Long - précision élevée'
        },
        'very_long': {
            'chunk_size': 900,
            'chunk_overlap': 120,
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
    if text_length < 2000:
        return configs['very_short']
    elif text_length < 5000:
        return configs['short']
    elif text_length < 10000:
        return configs['medium']
    elif text_length < 20000:
        return configs['long']
    else:
        return configs['very_long']

def extract_amadeus_sections(text, source_file):
    """
    Extrait les sections individuelles d'un document Amadeus
    pour un meilleur découpage sémantique.
    """
    # Détecter si c'est un document Amadeus
    if "Formats utiles Amadeus" not in text and "Formats et TUTO utiles" not in text:
        return None
    
    sections = []
    
    # Pattern pour détecter les sections Amadeus (format +- Titre)
    section_pattern = r'\+-([^\n]+)'
    
    # Trouver toutes les sections
    matches = list(re.finditer(section_pattern, text))
    
    if not matches:
        return None
    
    print(f"  ✓ Document Amadeus détecté avec {len(matches)} sections")
    
    # Extraire chaque section avec son contenu
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_pos = match.start()
        
        # Trouver la fin de la section (début de la section suivante ou fin du texte)
        if i < len(matches) - 1:
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        section_content = text[start_pos:end_pos].strip()
        
        # Enrichir avec le contexte du titre du document
        title_match = re.search(r'Titre du document : ([^\n]+)', text)
        doc_title = title_match.group(1) if title_match else "Formats Amadeus"
        
        enriched_content = f"Document: {doc_title}\nSection: {section_title}\n\n{section_content}"
        
        sections.append({
            'content': enriched_content,
            'title': section_title,
            'section_num': i + 1
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
            
            # Extraire texte + images
            text = extract_images_from_html(html_content, str(html_file))
            text_length = len(text)
            
            # Tenter d'extraire les sections Amadeus
            sections = extract_amadeus_sections(text, str(html_file))
            
            if sections:
                # Créer un document par section pour les documents Amadeus
                for section in sections:
                    section_length = len(section['content'])
                    chunk_config = get_adaptive_chunk_config(section_length, has_sections=True)
                    
                    doc = Document(
                        page_content=section['content'],
                        metadata={
                            "source": str(html_file),
                            "filename": html_file.name,
                            "type": "html_amadeus_section",
                            "section_title": section['title'],
                            "section_num": section['section_num'],
                            "chunk_size": chunk_config['chunk_size'],
                            "chunk_overlap": chunk_config['chunk_overlap'],
                            "text_length": section_length
                        }
                    )
                    documents.append(doc)
                    stats['amadeus_sections'] += 1
                
                print(f"✓ {html_file.name}: {len(sections)} sections Amadeus")
            else:
                # Traitement normal avec chunking adaptatif
                chunk_config = get_adaptive_chunk_config(text_length, has_sections=False)
                
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
                        "chunk_size": chunk_config['chunk_size'],
                        "chunk_overlap": chunk_config['chunk_overlap'],
                        "chunk_strategy": chunk_config['description'],
                        "text_length": text_length
                    }
                )
                documents.append(doc)
                
                print(f"✓ {html_file.name}: {text_length:,} chars → {chunk_config['description']} (chunk={chunk_config['chunk_size']})")
            
        except Exception as e:
            print(f"✗ Error loading {html_file.name}: {e}")
    
    print("="*80)
    print(f"\n📊 STATISTIQUES DE CHARGEMENT:")
    print(f"   • Sections Amadeus:     {stats['amadeus_sections']:3d}")
    print(f"   • Documents très courts: {stats['very_short']:3d} (chunk=400)")
    print(f"   • Documents courts:      {stats['short']:3d} (chunk=500)")
    print(f"   • Documents moyens:      {stats['medium']:3d} (chunk=600)")
    print(f"   • Documents longs:       {stats['long']:3d} (chunk=700)")
    print(f"   • Documents très longs:  {stats['very_long']:3d} (chunk=900)")
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
        "\n",              # Lignes simples
        " ",               # Mots
        "",                # Caractères
    ]
    
    # Appliquer le chunking par groupe
    for (chunk_size, chunk_overlap), docs in doc_groups.items():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            add_start_index=True,
        )
        
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)
        
        print(f"  • Chunk size {chunk_size}/{chunk_overlap}: {len(docs)} docs → {len(chunks)} chunks")
    
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
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
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
