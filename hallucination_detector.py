# hallucination_detector.py
"""
Hallucination detector ultra-léger sans ML
Utilise TF-IDF + semantic patterns
Compatible CPU-only
"""

from typing import List, Dict
from collections import Counter
import re
import math

class TFIDFMatcher:
    """Simple TF-IDF implementation for similarity checking"""
    
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.idf = {}
    
    def fit(self, documents: List[str]):
        """Build vocabulary and IDF"""
        self.documents = documents
        doc_freq = Counter()
        
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                doc_freq[word] += 1
            self.vocabulary.update(words)
        
        # Compute IDF
        n_docs = len(documents)
        for word, freq in doc_freq.items():
            self.idf[word] = math.log(n_docs / (freq + 1))
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) > 2]
    
    def _get_tfidf_vector(self, text: str) -> Dict[str, float]:
        """Get TF-IDF vector for text"""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        
        vector = {}
        total_terms = len(tokens) + 1  # Avoid division by zero
        
        for word, count in tf.items():
            if word in self.vocabulary:
                tfidf = (count / total_terms) * self.idf.get(word, 0)
                vector[word] = tfidf
        
        return vector
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts"""
        vec1 = self._get_tfidf_vector(text1)
        vec2 = self._get_tfidf_vector(text2)
        
        if not vec1 or not vec2:
            return 0.0
        
        # Compute dot product
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) 
                         for word in set(vec1.keys()) & set(vec2.keys()))
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


class HallucinationDetectorLightweight:
    """
    Hallucination detector SANS SentenceTransformer
    Utilise TF-IDF + pattern matching
    
    FIX: Seuil abaissé de 0.3 à 0.18 pour réduire les faux positifs
    sur les réponses paraphrasées mais correctes.
    """
    
    def __init__(self, threshold: float = 0.18):
        self.threshold = threshold
        self.tfidf = TFIDFMatcher()
        print("✓ HallucinationDetectorLightweight initialisé (CPU-only, seuil optimisé)")
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Regex pour phrases FR
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract main claims from text"""
        sentences = self.split_sentences(text)
        claims = []
        
        # Filter sentences that contain verbs/assertions
        assertion_patterns = [
            r'\b(est|sont|avoir|être|pouvoir|devoir|vouloir)\b',
            r'\b(existe|occupe|comprend|contient)\b',
            r'^\d+',  # Stats
        ]
        
        for sent in sentences:
            if any(re.search(pat, sent.lower()) for pat in assertion_patterns):
                claims.append(sent)
        
        return claims if claims else sentences
    
    def _word_overlap_score(self, claim: str, source_text: str) -> float:
        """
        Compute word overlap between claim and source (Jaccard-like).
        This catches paraphrases that TF-IDF might miss.
        """
        # Tokenize and filter stop words
        STOP_WORDS = {
            'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'au', 'aux',
            'ce', 'cette', 'ces', 'mon', 'ma', 'mes', 'ton', 'ta', 'tes',
            'son', 'sa', 'ses', 'notre', 'nos', 'votre', 'vos', 'leur', 'leurs',
            'je', 'tu', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles',
            'et', 'ou', 'mais', 'donc', 'car', 'ni', 'que', 'qui', 'quoi',
            'pour', 'par', 'avec', 'sans', 'sur', 'sous', 'dans', 'entre',
            'est', 'sont', 'être', 'avoir', 'fait', 'faire'
        }
        
        claim_words = set(w.lower() for w in re.findall(r'\b\w+\b', claim) 
                         if len(w) > 3 and w.lower() not in STOP_WORDS)
        source_words = set(w.lower() for w in re.findall(r'\b\w+\b', source_text) 
                          if len(w) > 3 and w.lower() not in STOP_WORDS)
        
        if not claim_words:
            return 0.0
        
        # Jaccard similarity
        intersection = claim_words & source_words
        union = claim_words | source_words
        
        if not union:
            return 0.0
            
        return len(intersection) / len(union)
    
    def check_hallucinations(self, response: str, source_documents: List[Dict]) -> Dict:
        """
        Check hallucinations in response using a multi-stage approach:
        1. Exact substring matching (high confidence)
        2. Sliding window TF-IDF (medium confidence)
        
        Returns:
            {
                'has_hallucinations': bool,
                'confidence_score': float (0-1),
                'hallucinated_claims': List[str],
                'supported_claims': List[str],
                'unclear_claims': List[str],
                'summary': str
            }
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
        
        # Extract claims from response
        response_claims = self.extract_claims(response)
        
        # Combine source text but keep track of chunks/sentences for granular matching
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
            
        # Prepare source chunks for sliding window TF-IDF
        # Split by sentences to avoid dilution
        source_sentences = self.split_sentences(full_source_text)
        # Create larger chunks (e.g., 3 sentences) for better context
        source_chunks = []
        for i in range(0, len(source_sentences), 1):
            chunk = " ".join(source_sentences[i:i+3])
            if len(chunk) > 50:
                source_chunks.append(chunk)
        
        # Also keep the full text for exact matching
        full_source_lower = full_source_text.lower()
        
        # Train TF-IDF on chunks (not the whole doc as one blob)
        if source_chunks:
            self.tfidf.fit(source_chunks)
        else:
            self.tfidf.fit([full_source_text])
        
        supported = []
        hallucinated = []
        unclear = []
        
        # Check each claim
        for claim in response_claims:
            try:
                claim_lower = claim.lower()
                
                # STAGE 1: Exact Substring Matching
                # If a significant part of the claim is verbatim in source, it's supported.
                if claim_lower in full_source_lower:
                    supported.append(claim)
                    continue
                
                # STAGE 2: Sliding Window TF-IDF
                # Compare claim against all source chunks and take the MAX similarity
                max_tfidf_sim = 0.0
                
                if source_chunks:
                    for chunk in source_chunks:
                        sim = self.tfidf.cosine_similarity(claim, chunk)
                        if sim > max_tfidf_sim:
                            max_tfidf_sim = sim
                else:
                    max_tfidf_sim = self.tfidf.cosine_similarity(claim, full_source_text)
                
                # STAGE 3: Word Overlap (for paraphrases)
                max_word_overlap = 0.0
                if source_chunks:
                    for chunk in source_chunks:
                        overlap = self._word_overlap_score(claim, chunk)
                        if overlap > max_word_overlap:
                            max_word_overlap = overlap
                else:
                    max_word_overlap = self._word_overlap_score(claim, full_source_text)
                
                # DECISION: Combine both metrics (FIX: seuils assouplis)
                # Le LLM paraphrase souvent les sources, ce qui est acceptable
                # On cherche à détecter les VRAIES hallucinations (info inventée)
                
                if max_tfidf_sim > self.threshold or max_word_overlap > 0.35:
                    # Evidence suffisante: TF-IDF OU word overlap acceptable
                    supported.append(claim)
                elif max_tfidf_sim < 0.08 and max_word_overlap < 0.15:
                    # Très faible sur les deux: probablement hallucination
                    hallucinated.append(claim)
                else:
                    # Borderline: considéré comme supporté (bénéfice du doute)
                    # FIX: avant, unclear comptait comme 0.5, maintenant comme supporté
                    supported.append(claim)
                    
            except Exception as e:
                print(f"⚠️  Error checking claim: {e}")
                unclear.append(claim)
        
        # Calculate confidence score (FIX: formule simplifiée)
        total = len(response_claims)
        if total == 0:
            confidence = 1.0
        else:
            # FIX: On compte uniquement les hallucinations avérées
            # Ratio = (total - hallucinated) / total
            # Si pas d'hallucination -> 1.0, si tout est halluciné -> 0.0
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
        """Add confidence badge to response"""
        confidence = check_result.get('confidence_score', 0.5)
        has_hallucinations = check_result.get('has_hallucinations', False)
        
        if confidence > 0.85 and not has_hallucinations:
            badge = "✅ **[HAUTE CONFIANCE]** Information validée dans les documents source."
        elif confidence > 0.65:
            badge = "⚠️ **[CONFIANCE MOYENNE]** Certaines informations n'ont pas de source directe - vérifiez auprès d'un expert."
        else:
            badge = "❌ **[BASSE CONFIANCE]** Consultez un expert RH ou Amadeus pour validation."
        
        return f"{response}\n\n{badge}"


# =============================================================================
# FACTORY FUNCTION - Sélection automatique CPU/GPU
# =============================================================================

def get_hallucination_detector(use_gpu: bool = None):
    """
    Factory pour obtenir le bon détecteur selon la configuration.
    
    Args:
        use_gpu: Force GPU (True) ou CPU (False). Si None, utilise la config.
    
    Returns:
        Instance de HallucinationDetectorLightweight (CPU) ou HallucinationDetectorGPU (GPU)
    """
    # Importer ici pour éviter import circulaire
    from device_config import USE_GPU as DEFAULT_USE_GPU
    
    should_use_gpu = use_gpu if use_gpu is not None else DEFAULT_USE_GPU
    
    if should_use_gpu:
        try:
            from hallucination_detector_gpu import HallucinationDetectorGPU
            return HallucinationDetectorGPU()
        except ImportError as e:
            print(f"⚠️ GPU detector non disponible ({e}), fallback CPU")
            return HallucinationDetectorLightweight()
        except Exception as e:
            print(f"⚠️ Erreur init GPU detector ({e}), fallback CPU")
            return HallucinationDetectorLightweight()
    else:
        return HallucinationDetectorLightweight()

