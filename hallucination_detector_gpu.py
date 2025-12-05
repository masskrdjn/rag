# hallucination_detector_gpu.py
"""
Hallucination Detector GPU avec SentenceTransformer.
Utilise des embeddings pour une détection plus précise des hallucinations.

Requiert: torch, sentence-transformers
"""

from typing import List, Dict
import re

# Import conditionnel
try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    import torch
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None

from device_config import get_device, USE_GPU


class HallucinationDetectorGPU:
    """
    Détecteur d'hallucinations utilisant SentenceTransformer sur GPU.
    
    Compare les embeddings des claims de la réponse avec les embeddings
    des sources pour détecter les informations non supportées.
    """
    
    # Modèle multilingue (supporte français)
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, model_name: str = None, threshold: float = 0.45):
        """
        Initialise le détecteur GPU.
        
        Args:
            model_name: Nom du modèle SentenceTransformer
            threshold: Seuil de similarité pour considérer un claim supporté
        """
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError(
                "sentence-transformers requis pour HallucinationDetectorGPU. "
                "Installez avec: pip install sentence-transformers"
            )
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.threshold = threshold
        self.device = get_device()
        
        print(f"🔄 Chargement SentenceTransformer: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print(f"✓ HallucinationDetectorGPU initialisé sur {self.device.upper()}")
    
    def split_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]
    
    def extract_claims(self, text: str) -> List[str]:
        """Extrait les claims/assertions du texte."""
        sentences = self.split_sentences(text)
        claims = []
        
        # Patterns d'assertion
        assertion_patterns = [
            r'\b(est|sont|avoir|être|pouvoir|devoir|vouloir)\b',
            r'\b(existe|occupe|comprend|contient)\b',
            r'^\d+',  # Stats
        ]
        
        for sent in sentences:
            if any(re.search(pat, sent.lower()) for pat in assertion_patterns):
                claims.append(sent)
        
        return claims if claims else sentences
    
    def check_hallucinations(self, response: str, source_documents: List[Dict]) -> Dict:
        """
        Vérifie les hallucinations en utilisant la similarité des embeddings.
        
        Args:
            response: Réponse du LLM
            source_documents: Documents sources
        
        Returns:
            Dict avec résultats de la vérification
        """
        if not source_documents:
            return {
                'has_hallucinations': False,
                'confidence_score': 1.0,
                'hallucinated_claims': [],
                'supported_claims': [],
                'unclear_claims': [],
                'reason': 'No source documents'
            }
        
        # Extraire claims de la réponse
        response_claims = self.extract_claims(response)
        
        if not response_claims:
            return {
                'has_hallucinations': False,
                'confidence_score': 1.0,
                'hallucinated_claims': [],
                'supported_claims': [],
                'unclear_claims': [],
                'reason': 'No claims in response'
            }
        
        # Combiner le texte source
        full_source_text = ' '.join([
            d.get('content', '') or d.get('page_content', '')
            for d in source_documents
        ])
        
        if not full_source_text.strip():
            return {
                'has_hallucinations': False,
                'confidence_score': 0.5,
                'hallucinated_claims': [],
                'supported_claims': [],
                'unclear_claims': response_claims,
                'reason': 'No source text content'
            }
        
        # Découper sources en phrases pour matching granulaire
        source_sentences = self.split_sentences(full_source_text)
        
        if not source_sentences:
            source_sentences = [full_source_text[:1000]]  # Fallback
        
        # Encoder les sources
        try:
            source_embeddings = self.model.encode(
                source_sentences,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"⚠️ Erreur encodage sources: {e}")
            return {
                'has_hallucinations': False,
                'confidence_score': 0.5,
                'hallucinated_claims': [],
                'supported_claims': [],
                'unclear_claims': response_claims,
                'reason': f'Encoding error: {e}'
            }
        
        supported = []
        hallucinated = []
        unclear = []
        
        # Vérifier chaque claim
        for claim in response_claims:
            try:
                # Encoder le claim
                claim_embedding = self.model.encode(
                    [claim],
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Calculer similarité avec toutes les sources
                similarities = cos_sim(claim_embedding, source_embeddings)[0]
                max_sim = float(similarities.max())
                
                if max_sim >= self.threshold:
                    supported.append(claim)
                elif max_sim < 0.25:
                    hallucinated.append(claim)
                else:
                    # Zone grise - bénéfice du doute
                    supported.append(claim)
                    
            except Exception as e:
                print(f"⚠️ Erreur vérification claim: {e}")
                unclear.append(claim)
        
        # Calculer score de confiance
        total = len(response_claims)
        if total == 0:
            confidence = 1.0
        else:
            confidence = (total - len(hallucinated)) / total
        
        return {
            'has_hallucinations': len(hallucinated) > 0,
            'confidence_score': confidence,
            'hallucinated_claims': hallucinated,
            'supported_claims': supported,
            'unclear_claims': unclear,
            'summary': f"{len(supported)}/{total} claims supported ({len(hallucinated)} rejected)"
        }
    
    def add_confidence_badge(self, response: str, check_result: Dict) -> str:
        """Ajoute un badge de confiance à la réponse."""
        confidence = check_result.get('confidence_score', 0.5)
        has_hallucinations = check_result.get('has_hallucinations', False)
        
        if confidence > 0.85 and not has_hallucinations:
            badge = "✅ **[HAUTE CONFIANCE]** Information validée dans les documents source."
        elif confidence > 0.65:
            badge = "⚠️ **[CONFIANCE MOYENNE]** Certaines informations n'ont pas de source directe."
        else:
            badge = "❌ **[BASSE CONFIANCE]** Consultez un expert pour validation."
        
        return f"{response}\n\n{badge}"


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_gpu_hallucination_detector(**kwargs) -> HallucinationDetectorGPU:
    """
    Factory pour créer un HallucinationDetectorGPU.
    
    Raises:
        ImportError: Si sentence-transformers non installé
    """
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        raise ImportError("sentence-transformers requis pour le détecteur GPU")
    
    if not USE_GPU:
        print("⚠️ GPU non activé, le détecteur utilisera CPU (peut être lent)")
    
    return HallucinationDetectorGPU(**kwargs)
