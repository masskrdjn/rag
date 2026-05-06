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
        print("HallucinationDetectorLightweight initialise (CPU-only)")
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        pieces = []
        for line in text.splitlines():
            line = re.sub(r'^\s*[-*]\s*', '', line).strip()
            if not line:
                continue
            pieces.extend(re.split(r'[.!?;]+', line))
        return [s.strip() for s in pieces if len(s.strip()) > 8]
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extrait les phrases factuelles du texte.

        On filtre les phrases purement neutres ou interrogatives (« Pour plus
        de détails... », « Voir ci-dessous ») qui ne portent pas d'assertion
        et gonflent le dénominateur du score de confiance. On conserve les
        items de liste (tirets) déjà nettoyés par split_sentences.
        """
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        assertion_patterns = [
            # Verbes d'état/action courants (présent, passé, futur, conditionnel)
            r'\b(est|sont|était|étaient|sera|seront|serait|seraient)\b',
            r'\b(a|ont|avait|avaient|aura|auront|aurait|auraient)\b',
            r'\b(peut|peuvent|doit|doivent|faut|veut|veulent)\b',
            r'\b(existe|comprend|contient|inclut|nécessite|permet|requiert)\b',
            r'\b(faire|émettre|annuler|modifier|consulter|envoyer|traiter)\b',
            # Phrase commençant par un nombre / chiffre / puce numérique
            r'^\s*\d+',
            # Items de liste avec : ou — qui définissent un terme
            r':\s*\S',
        ]

        claims = [
            sent for sent in sentences
            if any(re.search(pat, sent.lower()) for pat in assertion_patterns)
        ]

        # Si aucun pattern ne matche (cas des réponses très courtes), on garde
        # quand même tout pour ne pas masquer une éventuelle hallucination.
        return claims if claims else sentences

    # Petits entiers très courants en français (heures, jours, étapes,
    # numéros de section), trop bruités pour servir de proxy d'hallucination.
    _COMMON_SMALL_NUMBERS = {str(n) for n in range(0, 25)} | {
        "30", "60", "90", "100", "1000", "2024", "2025", "2026"
    }

    def _extract_sensitive_tokens(self, text: str) -> List[str]:
        """
        Extrait les tokens à fort signal factuel (dates, montants, codes IATA,
        identifiants) pour vérifier qu'ils sont bien présents dans la source.

        On ignore :
        - les citations [Sx] (ce sont des marqueurs internes, pas du contenu)
        - les mots simples avec majuscule initiale uniquement (« Galileo »,
          « Sabre ») qui ne sont pas des codes
        - les petits entiers communs qui apparaissent dans presque tous les
          textes et déclencheraient des faux positifs
        """
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b',          # dates JJ/MM[/AAAA]
            r'\b\d+(?:[.,]\d+)?\s?(?:%|eur|€|euros?)\b',        # montants
            r'\b\d{2,}\b',                                       # nombres ≥ 2 chiffres
            r'\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?\b',            # codes IATA / acronymes
        ]
        tokens = []
        for pattern in patterns:
            tokens.extend(re.findall(pattern, text, flags=re.IGNORECASE))
        citations = set(re.findall(r'\[S\d+\]', text, flags=re.IGNORECASE))
        filtered = []
        for token in dict.fromkeys(tokens):
            if token.upper() in citations or f"[{token.upper()}]" in citations:
                continue
            # On rejette les mots de type "Galileo" (majuscule initiale, pas un code)
            if (
                re.search(r'[A-Za-z]', token)
                and not re.search(r'\d', token)
                and token.upper() != token
            ):
                continue
            # On filtre les petits entiers courants (heures, étapes, années récentes)
            if token.isdigit() and token in self._COMMON_SMALL_NUMBERS:
                continue
            filtered.append(token)
        return filtered

    @staticmethod
    def _claim_citation_ids(claim: str) -> List[str]:
        return [m.upper().strip("[]") for m in re.findall(r'\[S\d+\]', claim, flags=re.IGNORECASE)]

    @staticmethod
    def _source_map(source_documents: List[Dict]) -> Dict[str, str]:
        mapped = {}
        for index, doc in enumerate(source_documents, 1):
            source_id = str(doc.get("source_id") or f"S{index}").upper()
            mapped[source_id] = doc.get('content', '') or doc.get('page_content', '')
        return mapped

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
        source_by_id = self._source_map(source_documents)
        
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
            
        # Préparer les chunks et matchers TF-IDF UNE SEULE FOIS, par source.
        # On évite ainsi de re-fitter à chaque claim (10× pour 10 claims).
        full_source_lower = full_source_text.lower()

        def _make_chunks(text: str) -> List[str]:
            sentences = self.split_sentences(text)
            chunks = []
            for idx in range(0, max(len(sentences), 1)):
                chunk = " ".join(sentences[idx:idx + 3])
                if len(chunk) > 30:
                    chunks.append(chunk)
            return chunks or [text]

        def _make_matcher(chunks: List[str]) -> TFIDFMatcher:
            matcher = TFIDFMatcher()
            matcher.fit(chunks)
            return matcher

        # Matcher global (toutes sources concaténées) pour les claims sans citation.
        global_chunks = _make_chunks(full_source_text)
        global_matcher = _make_matcher(global_chunks)

        # Matcher dédié par source_id, calculé à la demande puis caché.
        cached_matchers: Dict[tuple, tuple] = {}

        def _matcher_for_citations(citation_ids: List[str]):
            if not citation_ids:
                return global_matcher, global_chunks
            key = tuple(sorted(set(citation_ids)))
            if key not in cached_matchers:
                merged = " ".join(
                    source_by_id.get(sid, "") for sid in key
                ).strip()
                if not merged:
                    cached_matchers[key] = (None, [])
                else:
                    chunks = _make_chunks(merged)
                    cached_matchers[key] = (_make_matcher(chunks), chunks)
            return cached_matchers[key]

        supported = []
        hallucinated = []
        unclear = []

        # Check each claim
        for claim in response_claims:
            try:
                claim_lower = claim.lower()
                citation_ids = self._claim_citation_ids(claim)

                # STAGE 1: Exact Substring Matching
                # Si le claim est verbatim dans la source, considéré comme supporté.
                if claim_lower in full_source_lower:
                    if citation_ids:
                        supported.append(claim)
                    else:
                        unclear.append(f"{claim} [citation manquante]")
                    continue

                cited_source_text = " ".join(
                    source_by_id.get(source_id, "") for source_id in citation_ids
                ).strip()
                evidence_text = cited_source_text or full_source_text

                if citation_ids and not cited_source_text:
                    hallucinated.append(f"{claim} [citation inconnue]")
                    continue

                sensitive_tokens = self._extract_sensitive_tokens(claim)
                missing_sensitive = [
                    token for token in sensitive_tokens
                    if token.lower() not in evidence_text.lower()
                ]
                if missing_sensitive:
                    hallucinated.append(
                        f"{claim} [tokens absents: {', '.join(missing_sensitive[:5])}]"
                    )
                    continue

                matcher, comparison_chunks = _matcher_for_citations(citation_ids)
                if matcher is None or not comparison_chunks:
                    matcher, comparison_chunks = global_matcher, global_chunks

                # STAGE 2: Sliding Window TF-IDF — max sim sur les chunks
                max_tfidf_sim = 0.0
                for chunk in comparison_chunks:
                    sim = matcher.cosine_similarity(claim, chunk)
                    if sim > max_tfidf_sim:
                        max_tfidf_sim = sim

                # STAGE 3: Word Overlap (paraphrases)
                max_word_overlap = 0.0
                for chunk in comparison_chunks:
                    overlap = self._word_overlap_score(claim, chunk)
                    if overlap > max_word_overlap:
                        max_word_overlap = overlap
                
                # DECISION: Combine both metrics (FIX: seuils assouplis)
                # Le LLM paraphrase souvent les sources, ce qui est acceptable
                # On cherche à détecter les VRAIES hallucinations (info inventée)
                
                if max_tfidf_sim > self.threshold or max_word_overlap > 0.35:
                    # Evidence suffisante: TF-IDF OU word overlap acceptable
                    if citation_ids:
                        supported.append(claim)
                    else:
                        unclear.append(f"{claim} [citation manquante]")
                elif max_tfidf_sim < 0.08 and max_word_overlap < 0.15:
                    # Très faible sur les deux: probablement hallucination
                    hallucinated.append(claim)
                elif max_tfidf_sim <= self.threshold and max_word_overlap <= 0.35:
                    unclear.append(claim)
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
            penalty = (len(hallucinated) * 1.0) + (len(unclear) * 0.35)
            confidence = max(0.0, 1.0 - (penalty / total))
        
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

