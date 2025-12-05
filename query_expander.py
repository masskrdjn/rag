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
        """
        Thésaurus métier pour agents de voyages.
        Domaines: GDS (Amadeus/Galileo/Sabre), lowcost, train SNCF, billetterie.
        """
        return {
            # ══════════════════════════════════════════════════════════════
            # GDS - SYSTÈMES DE RÉSERVATION
            # ══════════════════════════════════════════════════════════════
            'gds': ['amadeus', 'galileo', 'sabre', 'système réservation', 'global distribution system', '1a', '1g', '1s'],
            'amadeus': ['gds', '1a', 'système amadeus', 'resa amadeus', 'cryptic'],
            'galileo': ['gds', '1g', 'travelport', 'système galileo'],
            'sabre': ['gds', '1s', 'système sabre'],
            '1a': ['amadeus', 'gds'],
            '1g': ['galileo', 'gds', 'travelport'],
            '1s': ['sabre', 'gds'],
            'cryptic': ['commandes', 'entrées', 'amadeus', 'format gds'],
            'cryptique': ['cryptic', 'commandes', 'entrées', 'amadeus', 'format gds'],
            
            # ══════════════════════════════════════════════════════════════
            # PNR / DOSSIER DE RÉSERVATION
            # ══════════════════════════════════════════════════════════════
            'pnr': ['dossier', 'réservation', 'booking', 'record locator', 'rloc'],
            'dossier': ['pnr', 'réservation', 'booking', 'case', 'demande'],
            'réservation': ['pnr', 'dossier', 'booking', 'resa'],
            'booking': ['pnr', 'réservation', 'dossier'],
            'segment': ['tronçon', 'vol', 'trajet', 'itinéraire', 'leg'],
            'itinéraire': ['trajet', 'parcours', 'segment', 'routing'],
            
            # ══════════════════════════════════════════════════════════════
            # ÉMISSION / BILLETTERIE
            # ══════════════════════════════════════════════════════════════
            'émission': ['ticketing', 'billet', 'émettre', 'création billet', 'tkt', 'etkt'],
            'émettre': ['émission', 'ticketing', 'créer billet', 'générer billet'],
            'billet': ['ticket', 'etkt', 'e-ticket', 'coupon', 'émission'],
            'ticket': ['billet', 'etkt', 'e-ticket', 'coupon'],
            'etkt': ['e-ticket', 'billet électronique', 'billet', 'ticket'],
            'ticketing': ['émission', 'billetterie', 'émettre'],
            'tst': ['transitional stored ticket', 'masque tarif', 'stockage tarif'],
            'void': ['annuler billet', 'voider', 'annulation émission'],
            'refund': ['remboursement', 'rembourser', 'rembs'],
            'remboursement': ['refund', 'rembourser', 'rembs', 'avoir'],
            
            # ══════════════════════════════════════════════════════════════
            # TARIFS / PRICING
            # ══════════════════════════════════════════════════════════════
            'tarif': ['prix', 'fare', 'pricing', 'montant', 'coût'],
            'fare': ['tarif', 'prix', 'pricing'],
            'pricing': ['tarification', 'calcul prix', 'tarif'],
            'fare basis': ['code tarif', 'classe tarifaire', 'base tarifaire'],
            'classe': ['cabine', 'classe réservation', 'rbd', 'booking class'],
            'rbd': ['classe', 'booking class', 'classe réservation'],
            'surcharge': ['taxe', 'supplément', 'yq', 'yr', 'fuel'],
            'taxe': ['surcharge', 'tax', 'frais'],
            'pénalité': ['frais', 'fee', 'penalty', 'noshow fee', 'frais modification'],
            
            # ══════════════════════════════════════════════════════════════
            # MODIFICATIONS / ÉCHANGES
            # ══════════════════════════════════════════════════════════════
            'échange': ['modification', 'changement', 'reissue', 'rerouting', 'rebooking'],
            'modification': ['échange', 'changement', 'modif', 'update'],
            'reissue': ['réémission', 'échange', 'nouvelle émission'],
            'rerouting': ['changement itinéraire', 'modification parcours', 'échange'],
            'annulation': ['cancel', 'void', 'suppression', 'no-show'],
            'no-show': ['non présentation', 'absence passager', 'noshow'],
            'revalidation': ['reval', 'prolongation', 'extension validité'],
            
            # ══════════════════════════════════════════════════════════════
            # COMPAGNIES / BSP
            # ══════════════════════════════════════════════════════════════
            'compagnie': ['carrier', 'transporteur', 'airline', 'cie aérienne'],
            'carrier': ['compagnie', 'transporteur', 'airline'],
            'bsp': ['billing settlement plan', 'règlement iata', 'facturation compagnie'],
            'iata': ['association', 'code iata', 'bsp', 'réglementation aérienne'],
            'interline': ['accord interline', 'interligne', 'correspondance compagnie'],
            'codeshare': ['partage code', 'vol partagé', 'operating carrier'],
            
            # ══════════════════════════════════════════════════════════════
            # LOWCOST
            # ══════════════════════════════════════════════════════════════
            'lowcost': ['low cost', 'bas coût', 'compagnie low cost', 'lcc'],
            'lcc': ['lowcost', 'low cost carrier', 'bas coût'],
            'bagage': ['bagages', 'luggage', 'bagage cabine', 'bagage soute', 'franchise'],
            'ancillary': ['service annexe', 'option', 'extra', 'add-on', 'prestation'],
            'option': ['ancillary', 'service', 'supplément', 'extra'],
            'siège': ['seat', 'place', 'attribution siège', 'seat map'],
            
            # ══════════════════════════════════════════════════════════════
            # TRAIN / SNCF
            # ══════════════════════════════════════════════════════════════
            'train': ['sncf', 'rail', 'ferroviaire', 'tgv', 'ter', 'ouigo', '77'],
            'sncf': ['train', 'rail', 'ferroviaire', '77', 'connecteur sncf'],
            '77': ['sncf', 'train', 'connecteur 77'],
            'tgv': ['train grande vitesse', 'sncf', 'train'],
            'ouigo': ['train lowcost', 'sncf lowcost', 'bas coût ferroviaire'],
            'gare': ['station', 'terminus', 'départ train'],
            
            # ══════════════════════════════════════════════════════════════
            # PASSAGERS / SSR
            # ══════════════════════════════════════════════════════════════
            'passager': ['pax', 'voyageur', 'client', 'passenger'],
            'pax': ['passager', 'voyageur', 'passenger', 'client'],
            'ssr': ['special service request', 'demande spéciale', 'service spécial'],
            'srr': ['special request', 'demande spéciale', 'ssr'],
            'repas': ['meal', 'nourriture', 'special meal', 'ssr meal'],
            'meal': ['repas', 'plateau repas', 'special meal'],
            'assistance': ['pmr', 'handicap', 'wheelchair', 'wchr', 'aide'],
            'infant': ['bébé', 'nourrisson', 'inf', 'lap child'],
            'enfant': ['child', 'chd', 'mineur', 'um', 'unaccompanied minor'],
            
            # ══════════════════════════════════════════════════════════════
            # QUEUE / WORKFLOW
            # ══════════════════════════════════════════════════════════════
            'queue': ['file attente', 'corbeille', 'todo', 'tâche'],
            'robot': ['automatique', 'auto', 'bot', 'automate', 'émission auto', 'traitement auto'],
            'alerte': ['warning', 'notification', 'message', 'erreur'],
            'time limit': ['délai', 'date limite', 'tl', 'ticketing time limit', 'ttl'],
            'ttl': ['time limit', 'délai émission', 'date limite'],
            
            # ══════════════════════════════════════════════════════════════
            # ERREURS / PROBLÈMES
            # ══════════════════════════════════════════════════════════════
            'échec': ['échoue', 'erreur', 'problème', 'failed', 'ko', 'impossible', 'reject'],
            'échoue': ['échec', 'erreur', 'problème', 'failed', 'ko'],
            'erreur': ['échec', 'échoue', 'problème', 'bug', 'failed', 'error'],
            'reject': ['rejet', 'refus', 'échec', 'ko'],
            'réclamation': ['plainte', 'litige', 'dispute', 'problème client', 'claim'],
            
            # ══════════════════════════════════════════════════════════════
            # INTERFACE / TECHNIQUE
            # ══════════════════════════════════════════════════════════════
            'connecteur': ['interface', 'système', 'api', 'connexion', 'middleware'],
            'api': ['interface', 'webservice', 'ws', 'connexion', 'intégration'],
            'couleur': ['code couleur', 'statut', 'indicateur', 'signification', 'légende'],
            'format': ['structure', 'modèle', 'template', 'gabarit', 'masque'],
            'procédure': ['étapes', 'processus', 'démarche', 'workflow', 'mode opératoire'],
            'étape': ['phase', 'step', 'niveau', 'action'],
            
            # ══════════════════════════════════════════════════════════════
            # VOYAGE LOISIRS / PARTICULIERS
            # ══════════════════════════════════════════════════════════════
            'vacances': ['séjour', 'voyage', 'holidays', 'congés', 'loisirs'],
            'séjour': ['vacances', 'voyage', 'package', 'forfait', 'hébergement'],
            'package': ['forfait', 'séjour', 'tout compris', 'all inclusive'],
            'forfait': ['package', 'séjour', 'formule'],
            'hôtel': ['hébergement', 'hotel', 'logement', 'accommodation', 'nuitée'],
            'hébergement': ['hôtel', 'logement', 'residence', 'location'],
            'destination': ['pays', 'ville', 'lieu', 'arrivée'],
            'tourisme': ['loisirs', 'vacances', 'voyage agrément'],
            'location': ['voiture', 'car rental', 'véhicule', 'rent a car'],
            'transfert': ['navette', 'shuttle', 'transport aéroport'],
            'assurance': ['insurance', 'annulation', 'rapatriement', 'couverture'],
            'visa': ['passeport', 'document voyage', 'autorisation', 'esta', 'eta'],
            'famille': ['enfant', 'bébé', 'groupe', 'familial'],
            'groupe': ['collectif', 'tour', 'circuit', 'voyage organisé'],
            'circuit': ['tour', 'itinéraire', 'voyage organisé', 'excursion'],
            'excursion': ['visite', 'activité', 'sortie', 'tour'],
            
            # ══════════════════════════════════════════════════════════════
            # RH / ABSENCES (contexte interne)
            # ══════════════════════════════════════════════════════════════
            'absence': ['congé', 'jour', 'vacance', 'permission', 'indisponibilité'],
            'congé': ['absence', 'jour off', 'vacance', 'permission', 'cp'],
            'maladie': ['arrêt', 'repos', 'indisponibilité', 'arrêt maladie'],

            # ══════════════════════════════════════════════════════════════
            # OUTILS INTERNES (à adapter selon l'entreprise)
            # ══════════════════════════════════════════════════════════════
            'whaller': ['intranet', 'sphère', 'réseau interne', 'documentation interne'],
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
