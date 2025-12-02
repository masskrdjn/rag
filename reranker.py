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


class RAGRerankerLightweight:
    """
    Reranker multi-critères SANS ML - CPU optimisé
    Combinaison de:
    - BM25 (Okapi ranking)
    - Keyword overlap
    - Metadata scoring
    """
    
    def __init__(self):
        self.bm25 = None
        self.stop_words_fr = {
            'le', 'la', 'de', 'des', 'et', 'ou', 'est', 'un', 'une',
            'à', 'au', 'en', 'dans', 'pour', 'par', 'sur', 'avec', 'sans',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
            'comment', 'où', 'quand', 'pourquoi', 'qui', 'quoi'
        }
        print("✓ RAGRerankerLightweight initialisé (CPU-only, pas de GPU requis)")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords (>3 chars, non-stopwords)"""
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in self.stop_words_fr]
        return keywords
    
    def calculate_keyword_overlap(self, question_keywords: List[str], doc_content: str) -> float:
        """Calculate keyword coverage percentage"""
        doc_keywords = set(self.extract_keywords(doc_content))
        if not question_keywords:
            return 0.0
        
        overlap = sum(1 for kw in question_keywords if kw in doc_keywords)
        coverage = overlap / len(question_keywords)
        return min(coverage, 1.0)
    
    def _score_metadata(self, document: Dict, question: str) -> float:
        """Score metadata/reliability"""
        score = 0.7
        
        # Bonus for summaries with general questions
        is_summary = document.get('is_summary', False)
        if is_summary:
            general_keywords = {'comment', 'quoi', 'qu', 'général', 'résumé'}
            if any(kw in question.lower() for kw in general_keywords):
                score = 0.9
            else:
                score = 0.5
        
        # Penalty for redirect documents
        content = document.get('content', '') or document.get('page_content', '')
        if 'consulter' in content.lower() and 'whaller' in content.lower():
            score *= 0.6
        
        # Bonus for reliability score
        if 'reliability_score' in document:
            score = (score + document['reliability_score']) / 2
        
        return min(score, 1.0)
    
    def score_document(self, question: str, document: Dict, 
                      base_similarity: float = 0.5) -> float:
        """
        Composite score combining multiple signals
        """
        scores = {}
        content = document.get('content', '') or document.get('page_content', '')
        
        # 1. Base similarity score (from ChromaDB vector search)
        scores['semantic'] = min(base_similarity, 1.0)
        
        # 2. BM25 score (no neural network required)
        if content:
            bm25_score = BM25([content]).score_query(question, 0)
            scores['bm25'] = min(bm25_score / 10.0, 1.0)  # Normalize
        else:
            scores['bm25'] = 0.0
        
        # 3. Keyword matching score
        question_keywords = self.extract_keywords(question)
        scores['keyword_match'] = self.calculate_keyword_overlap(question_keywords, content)
        
        # 4. Metadata score
        scores['metadata'] = self._score_metadata(document, question)
        
        # Optimized weights (no ML component)
        weights = {
            'semantic': 0.40,        # Increased (was 0.30)
            'bm25': 0.30,            # Replaces cross-encoder
            'keyword_match': 0.20,
            'metadata': 0.10
        }
        
        # Composite score
        final_score = sum(scores.get(k, 0) * weights[k] for k in weights.keys())
        return final_score
    
    def rerank(self, question: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank and filter results
        
        Args:
            question: User question
            documents: List of documents with similarity_score
            top_k: Number of results to return
        
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        # Score each document
        scored = []
        for doc in documents:
            base_sim = doc.get('similarity_score', 0.5)
            rerank_score = self.score_document(question, doc, base_sim)
            scored.append({
                **doc,
                'rerank_score': rerank_score
            })
        
        # Sort by composite score
        sorted_docs = sorted(scored, key=lambda x: x['rerank_score'], reverse=True)
        
        # Filter out low scores
        filtered = [d for d in sorted_docs if d['rerank_score'] > 0.20]
        
        # Return top-k
        return filtered[:top_k]
