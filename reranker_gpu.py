# reranker_gpu.py
"""
Reranker GPU avec CrossEncoder pour un reranking de haute qualité.
Utilise le modèle ms-marco-MiniLM pour une recherche sémantique précise.

Requiert: torch, sentence-transformers
"""

from typing import List, Dict
import re

# Import conditionnel pour éviter erreur si GPU non dispo
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None

from device_config import get_device, USE_GPU


class RAGRerankerGPU:
    """
    Reranker haute performance utilisant CrossEncoder sur GPU.
    
    Le CrossEncoder score directement les paires (question, document)
    pour un ranking plus précis que les embeddings séparés.
    """
    
    # Modèle recommandé: léger et performant
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(self, model_name: str = None, batch_size: int = 16):
        """
        Initialise le reranker GPU.
        
        Args:
            model_name: Nom du modèle CrossEncoder (HuggingFace)
            batch_size: Taille du batch pour l'inférence
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers requis pour RAGRerankerGPU. "
                "Installez avec: pip install sentence-transformers"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.device = get_device()
        
        print(f"🔄 Chargement CrossEncoder: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=self.device
        )
        print(f"✓ RAGRerankerGPU initialisé sur {self.device.upper()}")
        
        # Stop words pour keyword matching (backup)
        self.stop_words_fr = {
            'le', 'la', 'de', 'des', 'et', 'ou', 'est', 'un', 'une',
            'à', 'au', 'en', 'dans', 'pour', 'par', 'sur', 'avec', 'sans',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
            'comment', 'où', 'quand', 'pourquoi', 'qui', 'quoi'
        }
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extrait les mots-clés significatifs."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) > 3 and w not in self.stop_words_fr]
    
    def _score_metadata(self, document: Dict, question: str) -> float:
        """Score basé sur les métadonnées (bonus/malus)."""
        score_modifier = 0.0
        
        # Bonus si titre du document matche
        filename = document.get('filename', '').lower().replace('_', ' ').replace('.html', '')
        question_keywords = self.extract_keywords(question)
        title_matches = sum(1 for kw in question_keywords if kw in filename)
        
        if title_matches >= 2:
            score_modifier += 0.1
        elif title_matches == 1:
            score_modifier += 0.05
        
        # Malus pour documents très courts (redirects)
        content = document.get('content', '') or document.get('page_content', '')
        if len(content) < 200:
            score_modifier -= 0.15
        
        return score_modifier
    
    def score_document(self, question: str, document: Dict,
                       base_similarity: float = 0.5) -> float:
        """
        Score un document avec CrossEncoder.
        
        Args:
            question: Question utilisateur
            document: Document avec 'content' ou 'page_content'
            base_similarity: Score de similarité vectorielle (pour backup)
        
        Returns:
            Score normalisé entre 0 et 1
        """
        content = document.get('content', '') or document.get('page_content', '')
        
        if not content.strip():
            return 0.0
        
        # Tronquer le contenu si trop long (CrossEncoder a une limite)
        if len(content) > 1500:
            content = content[:1500]
        
        # Score CrossEncoder
        try:
            cross_score = self.model.predict([(question, content)])[0]
            # Normaliser entre 0 et 1 (les scores CrossEncoder peuvent être négatifs)
            normalized_score = (cross_score + 10) / 20  # Approximation
            normalized_score = max(0.0, min(1.0, normalized_score))
        except Exception as e:
            print(f"⚠️ Erreur CrossEncoder, fallback similarité: {e}")
            normalized_score = base_similarity
        
        # Ajouter bonus/malus métadonnées
        metadata_modifier = self._score_metadata(document, question)
        final_score = normalized_score + metadata_modifier
        
        return max(0.0, min(1.0, final_score))
    
    def rerank(self, question: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank les documents avec CrossEncoder.
        
        Args:
            question: Question utilisateur
            documents: Liste de documents avec similarity_score
            top_k: Nombre de résultats à retourner
        
        Returns:
            Documents reranked et filtrés
        """
        if not documents:
            return []
        
        # Préparer les paires pour scoring batch
        contents = []
        for doc in documents:
            content = doc.get('content', '') or doc.get('page_content', '')
            # Tronquer si nécessaire
            if len(content) > 1500:
                content = content[:1500]
            contents.append(content)
        
        # Score en batch avec CrossEncoder
        pairs = [(question, content) for content in contents]
        
        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size)
        except Exception as e:
            print(f"⚠️ Erreur batch CrossEncoder: {e}")
            # Fallback vers scoring individuel
            scores = [self.score_document(question, doc) for doc in documents]
        
        # Construire résultats avec scores normalisés
        scored = []
        for i, doc in enumerate(documents):
            cross_score = scores[i] if isinstance(scores[i], float) else float(scores[i])
            # Normaliser
            normalized = (cross_score + 10) / 20
            normalized = max(0.0, min(1.0, normalized))
            
            # Ajouter metadata modifier
            metadata_mod = self._score_metadata(doc, question)
            final_score = max(0.0, min(1.0, normalized + metadata_mod))
            
            scored.append({
                **doc,
                'rerank_score': final_score,
                'cross_encoder_raw': cross_score
            })
        
        # Trier par score
        sorted_docs = sorted(scored, key=lambda x: x['rerank_score'], reverse=True)
        
        # Filtrer les scores très bas
        min_threshold = 0.25
        filtered = [d for d in sorted_docs if d['rerank_score'] > min_threshold]
        
        # Fallback si tout filtré
        if not filtered and sorted_docs:
            filtered = sorted_docs[:1]
        
        return filtered[:top_k]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_gpu_reranker(**kwargs) -> RAGRerankerGPU:
    """
    Factory pour créer un RAGRerankerGPU.
    
    Raises:
        ImportError: Si sentence-transformers non installé
        RuntimeError: Si GPU requis mais non disponible
    """
    if not CROSS_ENCODER_AVAILABLE:
        raise ImportError("sentence-transformers requis pour le reranker GPU")
    
    if not USE_GPU:
        print("⚠️ GPU non activé, le reranker utilisera CPU (peut être lent)")
    
    return RAGRerankerGPU(**kwargs)
