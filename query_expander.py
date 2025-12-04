# query_expander.py
from typing import List, Set
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryExpander:
    """
    Expander de requête pour améliorer la couverture de recherche.
    
    Techniques:
    - Expansion LLM (variantes sémantiques)
    - Expansion synonymes (thésaurus métier)
    - Expansion morphologique (singulier/pluriel)
    """
    
    def __init__(self, model_name: str = "mistral:7b"):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=0.7)
        self.synonyms_map = self._build_synonyms_map()
    
    def _build_synonyms_map(self) -> dict:
        """Construire un thésaurus métier"""
        return {
            'absence': ['congé', 'jour', 'vacance', 'permission'],
            'congé': ['absence', 'jour', 'vacance', 'permission'],
            'maladie': ['arrêt', 'repos', 'indisponibilité'],
            'format': ['structure', 'modèle', 'template', 'gabarit'],
            'amadeus': ['système', 'gds', 'booking', 'système de réservation', '1a'],
            'émission': ['création', 'génération', 'production', 'émettre', 'ticketing', 'billet'],
            'dossier': ['ticket', 'case', 'demande', 'requête', 'pnr'],
            'procédure': ['étapes', 'processus', 'démarche', 'étapes'],
            'étape': ['phase', 'étapes', 'niveau'],
            'utiliser': ['employer', 'utilisation', 'usage', 'use'],
            'spécial': ['spéciaux', 'particulier', 'particuliers'],
            'repas': ['nourriture', 'meal', 'service'],
            'srr': ['special request', 'demande spéciale', 'special meal'],
            # Ajouts pour améliorer le retrieval
            'échec': ['échoue', 'erreur', 'problème', 'failed', 'ko', 'impossible'],
            'échoue': ['échec', 'erreur', 'problème', 'failed', 'ko'],
            'erreur': ['échec', 'échoue', 'problème', 'bug', 'failed'],
            'robot': ['automatique', 'auto', 'bot', 'automate', 'émission auto', 'logo robot'],
            'réclamation': ['plainte', 'litige', 'problème client'],
            'échange': ['modification', 'changement', 'rerouting'],
            'bsp': ['compagnie', 'iata', 'aérien'],
            # Nouveaux ajouts pour GDS et connecteurs
            'gds': ['galileo', 'sabre', 'amadeus', 'système réservation', '1g', '1s', '1a'],
            'galileo': ['gds', '1g', 'système réservation'],
            'sabre': ['gds', '1s', 'système réservation'],
            'connecteur': ['interface', 'système', 'api', 'connexion', 'connectivité'],
            'sncf': ['train', 'rail', '77', 'ferroviaire'],
            'train': ['sncf', 'rail', 'ferroviaire', '77'],
            'iata': ['bsp', 'compagnie', 'resa iata', 'émission iata'],
            'couleur': ['code couleur', 'statut', 'indicateur', 'codes', 'signification', 'logo'],
            'émettre': ['émission', 'ticketing', 'billet', 'création'],
        }
    
    def expand_with_synonyms(self, question: str) -> Set[str]:
        """Générer des variantes avec synonymes"""
        expanded = {question}
        question_lower = question.lower()
        
        for term, alternatives in self.synonyms_map.items():
            if term in question_lower:
                for alt in alternatives:
                    # Remplacer le terme
                    variant = question_lower.replace(term, alt)
                    expanded.add(variant)
                    
                    # Aussi ajouter en capitalisé
                    expanded.add(variant.capitalize())
        
        return expanded
    
    def expand_morphological(self, question: str) -> Set[str]:
        """Générer variantes singulier/pluriel"""
        expanded = {question}
        
        # Règles simples pluriel FR
        plural_rules = [
            (r'(\w+)$', lambda m: m.group(1) + 's'),  # Suffixe général
            (r'(\w+)tion$', lambda m: m.group(1) + 'tions'),
            (r'(\w+)ement$', lambda m: m.group(1) + 'ements'),
        ]
        
        for pattern, rule in plural_rules:
            matches = re.finditer(pattern, question.lower())
            for match in matches:
                variant = rule(match)
                if len(variant) > 3:  # Ignorer les mots courts
                    expanded.add(question.lower().replace(match.group(0), variant))
        
        return expanded
    
    def expand_with_llm(self, question: str, num_variants: int = 2) -> Set[str]:
        """Générer variantes avec LLM (coûteux mais meilleur)"""
        expanded = {question}
        
        template = """Génère {num_variants} variantes de cette question pour améliorer la recherche dans une base documentaire de procédures de voyage.

CONSIGNES :
- Si la question est vague, génère des variantes SPÉCIFIQUES explorant différents cas (ex: "échec émission" → "échec émission GDS", "échec émission train")
- Si la question mentionne un élé technique, génère des variantes avec des termes connexes
- Garde les variantes COURTES (max 15 mots)
- En français uniquement

Question: "{question}"

Variantes (une par ligne, sans numérotation):"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"num_variants": num_variants, "question": question})
            
            variants = [
                v.strip() 
                for v in response.strip().split('\n') 
                if v.strip() and len(v.strip()) > 5
            ]
            
            expanded.update(variants[:num_variants])
            
        except Exception as e:
            print(f"⚠️ LLM expansion échouée: {e}")
        
        return expanded
    
    def get_all_variants(self, question: str, 
                        use_llm: bool = False,
                        llm_variants: int = 2) -> List[str]:
        """Récupérer toutes les variantes de la question"""
        
        all_variants = {question}
        
        # Synonymes (rapide, toujours faire)
        all_variants.update(self.expand_with_synonyms(question))
        
        # Morphologie (rapide, toujours faire)
        all_variants.update(self.expand_morphological(question))
        
        # LLM (coûteux, optionnel)
        if use_llm:
            all_variants.update(
                self.expand_with_llm(question, llm_variants)
            )
        
        return sorted(list(all_variants))
