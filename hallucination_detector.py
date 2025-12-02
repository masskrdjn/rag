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
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.tfidf = TFIDFMatcher()
        print("✓ HallucinationDetectorLightweight initialisé (CPU-only, pas de dépendances ML)")
    
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
    
    def check_hallucinations(self, response: str, source_documents: List[Dict]) -> Dict:
        """
        Check hallucinations in response
        
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
        
        # Combine source text
        source_text = ' '.join([
            d.get('content', '') or d.get('page_content', '') 
            for d in source_documents
        ])
        
        if not source_text.strip():
            return {
                'has_hallucinations': False,
                'confidence_score': 0.5,
                'hallucinated_claims': [],
                'supported_claims': [],
                'unclear_claims': response_claims,
                'reason': 'No source text content'
            }
        
        # Train TF-IDF on sources
        self.tfidf.fit([source_text])
        
        supported = []
        hallucinated = []
        unclear = []
        
        # Check each claim
        for claim in response_claims:
            try:
                similarity = self.tfidf.cosine_similarity(claim, source_text)
                
                if similarity > self.threshold + 0.2:
                    supported.append(claim)
                elif similarity < self.threshold - 0.1:
                    hallucinated.append(claim)
                else:
                    unclear.append(claim)
            except Exception as e:
                print(f"⚠️  Error checking claim: {e}")
                unclear.append(claim)
        
        # Calculate confidence score
        total = len(response_claims)
        confidence = len(supported) / max(total, 1)
        
        return {
            'has_hallucinations': len(hallucinated) > 0,
            'confidence_score': confidence,
            'hallucinated_claims': hallucinated,
            'supported_claims': supported,
            'unclear_claims': unclear,
            'summary': f"{len(supported)}/{total} claims supported"
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
