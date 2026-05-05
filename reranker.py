# reranker.py
"""
Reranker ultra-léger sans ML - CPU optimisé
Compatible avec configurations GPU limitées
Technique: BM25 + Keyword overlap + Metadata scoring
"""

from typing import List, Dict
from collections import Counter
import re
import math
import unicodedata

from config import RAG_CONFIG


KEYWORD_TAXONOMY = RAG_CONFIG["keyword_taxonomy"]
BROAD_KEYWORDS = tuple(KEYWORD_TAXONOMY.get("broad", ()))
SPECIFIC_KEYWORDS = tuple(KEYWORD_TAXONOMY.get("specific", ()))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def contains_keyword(text: str, keywords) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(kw) in normalized for kw in keywords)

class BM25:
    """Implémentation BM25 (Okapi) - sans dépendances externes"""
    
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1  # Control term frequency saturation point
        self.b = b    # Control how much effect doc length has on relevance
        
        self.corpus_size = len(documents)
        self.avgdl = sum(len(self._tokenize(doc)) for doc in documents) / max(self.corpus_size, 1)
        
        # Build inverted index
        self.idf = {}
        self._compute_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize et normalize (French-aware)"""
        words = re.findall(r'\b\w+\b', text.lower())
        # Simple stopwords
        stop_words = {'le', 'la', 'de', 'des', 'et', 'ou', 'est', 'un', 'une', 'à', 'au', 'en', 'dans', 'pour', 'par', 'sur', 'avec', 'sans', 'ce', 'cette', 'ces'}
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _compute_idf(self):
        """Compute IDF for all terms"""
        doc_counts = Counter()
        for doc in self.documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_counts[token] += 1
        
        for term, count in doc_counts.items():
            self.idf[term] = math.log(
                (self.corpus_size - count + 0.5) / (count + 0.5) + 1.0
            )
    
    def score_query(self, query: str, doc_index: int) -> float:
        """Score a document for a query"""
        query_tokens = self._tokenize(query)
        doc = self.documents[doc_index]
        doc_tokens = self._tokenize(doc)
        doc_len = len(doc_tokens)
        
        score = 0.0
        doc_counter = Counter(doc_tokens)
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            idf_score = self.idf[token]
            freq = doc_counter.get(token, 0)
            
            # BM25 formula
            numerator = idf_score * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            
            score += numerator / denominator
        
        return score


class RAGRerankerStrict:
    """
    Reranker strict sans ML - tuné pour CPU + BM25 retrieval
    Scoring agressif pour rejeter le bruit
    """
    
    def __init__(self):
        self.stop_words_fr = {
            'le', 'la', 'de', 'des', 'et', 'ou', 'est', 'un', 'une',
            'à', 'au', 'en', 'dans', 'pour', 'par', 'sur', 'avec', 'sans',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
            'comment', 'où', 'quand', 'pourquoi', 'qui', 'quoi'
        }
        print("RAGRerankerStrict initialise (CPU-only, scoring strict)")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords (>3 chars, non-stopwords)"""
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in self.stop_words_fr]
        return keywords
    
    def calculate_keyword_overlap(self, keywords: List[str], content: str) -> float:
        """Calculate keyword coverage"""
        if not keywords:
            return 0.0
        content_lower = content.lower()
        overlap = sum(1 for kw in keywords if kw in content_lower)
        return min(overlap / len(keywords), 1.0)
    
    def _score_metadata(self, document: Dict, question: str) -> float:
        """Metadata scoring with title matching boost"""
        score = 0.5
        
        # NOUVEAU: Bonus si le titre du document match la question
        filename = normalize_text(document.get('filename', '').replace('_', ' ').replace('.html', ''))
        question_lower = normalize_text(question)
        question_keywords = self.extract_keywords(question)
        
        # Compter les mots-clés de la question présents dans le filename
        title_matches = sum(1 for kw in question_keywords if kw in filename)
        if title_matches >= 2:
            score = 0.9  # Fort bonus si 2+ mots-clés matchent le titre
        elif title_matches == 1:
            score = 0.7
        
        # Bonus pour documents avec structure (sections +-))
        if document.get('is_summary', False):
            if any(kw in question_lower for kw in ['comment', 'quoi', 'résumé']):
                score = max(score, 0.8)
            else:
                score = max(score, 0.4)
        
        # Pénalité pour redirects trop court
        is_summary = bool(document.get('is_summary', False))
        is_broad = contains_keyword(question, BROAD_KEYWORDS)
        is_specific = contains_keyword(question, SPECIFIC_KEYWORDS)
        if is_summary and is_broad and not is_specific:
            score = max(score, 0.85)
        elif is_summary and is_specific:
            score = min(score, 0.35)

        section_title = normalize_text(document.get('section_title', ''))
        section_matches = sum(
            1 for kw in question_keywords if normalize_text(kw) in section_title
        )
        if section_matches:
            score = min(1.0, score + min(section_matches * 0.08, 0.20))

        try:
            reliability = max(0.0, min(float(document.get('reliability_score', 0.5)), 1.0))
            score = (score * 0.85) + (reliability * 0.15)
        except (TypeError, ValueError):
            pass

        try:
            density = float(document.get('information_density', 0.5))
            if density < 0.18:
                score *= 0.9
            elif density > 0.45:
                score = min(1.0, score + 0.05)
        except (TypeError, ValueError):
            pass

        content = document.get('content', '')
        if len(content) < 200 and 'whaller' in content.lower():
            score *= 0.6
        
        return min(score, 1.0)
    
    def score_document(self, question: str, document: Dict,
                      base_similarity: float = 0.5) -> float:
        """
        Scoring strict avec gates agressifs + boost keyword exact matches
        """
        content = document.get('content', '') or document.get('page_content', '')
        content_lower = normalize_text(content)
        question_lower = normalize_text(question)
        
        # 🔴 GATE 1: Rejeter contenu très court (< 150 chars)
        if len(content.strip()) < 150:
            is_redirect = 'consulter' in content_lower and 'whaller' in content_lower
            # FIX: Accept Whaller redirects if similarity is decent
            if is_redirect and base_similarity > 0.30:  # Abaissé de 0.65 à 0.30
                return 0.60  # Boost pour Whaller redirects
            else:
                return 0.15
        
        scores = {}
        
        # 1. Similarité vectorielle (PRIME)
        scores['semantic'] = min(base_similarity, 1.0)
        
        # 🔴 GATE 2: Rejeter base_sim très faible
        # FIX: Check for exact keyword matches BEFORE rejecting
        question_keywords = self.extract_keywords(question)
        exact_matches = sum(1 for kw in question_keywords if kw in content_lower)
        
        if base_similarity < 0.30 and exact_matches < 2:
            return 0.10
        
        # 2. Longueur du contenu
        content_norm = min(len(content) / 4000.0, 1.0)
        scores['content_length'] = content_norm
        
        # 3. Densité de sentences (structure)
        num_sentences = max(len(re.split(r'[.!?]', content)), 1)
        structure_score = min(num_sentences / 15.0, 1.0)
        scores['structure'] = structure_score
        
        # 4. Keyword overlap avec BOOST pour matches exacts multiples
        keyword_score = self.calculate_keyword_overlap(question_keywords, content)
        
        # 🆕 BOOST: Si 2+ mots-clés exacts matchent (ex: "couleurs" + "robot")
        if exact_matches >= 2:
            keyword_score = min(keyword_score * 1.5, 1.0)
        
        scores['keyword_match'] = keyword_score
        
        # 5. Métadonnées
        scores['metadata'] = self._score_metadata(document, question)
        
        # Poids avec augmentation du keyword matching
        weights = {
            'semantic': 0.50,        # Légèrement réduit
            'content_length': 0.12,
            'structure': 0.08,
            'keyword_match': 0.20,   # AUGMENTÉ de 0.10 à 0.20
            'metadata': 0.10
        }
        
        final = sum(scores.get(k, 0) * weights[k] for k in weights.keys())
        return final
    
    def rerank(self, question: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank avec filtrage adaptatif selon la question
        Threshold dynamique: questions précises = strict, questions larges = permissif
        """
        if not documents:
            return []
        
        # Détecter questions larges/vagues nécessitant plus de contexte
        question_lower = normalize_text(question)
        vague_keywords = ['comment', 'que faire', 'quels', 'quelles', 'quel']
        is_vague = any(kw in question_lower for kw in vague_keywords) or contains_keyword(question, BROAD_KEYWORDS)
        
        # Score tous les docs
        scored = []
        for doc in documents:
            base_sim = doc.get('similarity_score', 0.5)
            rerank_score = self.score_document(question, doc, base_sim)
            
            scored.append({
                **doc,
                'rerank_score': rerank_score
            })
        
        # Trier par score
        sorted_docs = sorted(
            scored,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        # 🆕 Filtrage ADAPTATIF:
        # - Questions vagues: threshold bas (0.35) pour inclure plus de contexte
        # - Questions précises: threshold strict (0.40)
        if is_vague:
            min_score = 0.32  # Plus permissif pour questions larges
        else:
            min_score = 0.38  # Légèrement abaissé de 0.40
        
        filtered = [d for d in sorted_docs if d['rerank_score'] > min_score]
        
        # Fallback: si < 2 résultats, abaisser encore
        if len(filtered) < 2 and len(sorted_docs) > 0:
            min_score = 0.28 if is_vague else 0.32
            filtered = [d for d in sorted_docs if d['rerank_score'] > min_score]
        
        # Jamais 0 résultats
        if len(filtered) == 0:
            filtered = sorted_docs[:1]
        
        return filtered[:top_k]


# =============================================================================
# FACTORY FUNCTION - Sélection automatique CPU/GPU
# =============================================================================

def get_reranker(use_gpu: bool = None):
    """
    Factory pour obtenir le bon reranker selon la configuration.
    
    Args:
        use_gpu: Force GPU (True) ou CPU (False). Si None, utilise la config.
    
    Returns:
        Instance de RAGRerankerStrict (CPU) ou RAGRerankerGPU (GPU)
    """
    # Importer ici pour éviter import circulaire
    from device_config import USE_GPU as DEFAULT_USE_GPU
    
    should_use_gpu = use_gpu if use_gpu is not None else DEFAULT_USE_GPU
    
    if should_use_gpu:
        try:
            from reranker_gpu import RAGRerankerGPU
            return RAGRerankerGPU()
        except ImportError as e:
            print(f"⚠️ GPU reranker non disponible ({e}), fallback CPU")
            return RAGRerankerStrict()
        except Exception as e:
            print(f"⚠️ Erreur init GPU reranker ({e}), fallback CPU")
            return RAGRerankerStrict()
    else:
        return RAGRerankerStrict()

