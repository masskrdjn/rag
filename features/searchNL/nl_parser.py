# nl_parser.py
"""
Analyseur de langage naturel pour les requêtes de recherche de vols.
Utilise qwen-14b via Ollama pour extraire des informations de vol structurées.
"""
import json
import re
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "nl_parser.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from .config_searchnl import (
    MODEL_NAME, OLLAMA_CONFIG, SYSTEM_PROMPT, EXTRACTION_PROMPT_TEMPLATE
)
from .airport_db import AirportDatabase
from .date_parser import DateParser


@dataclass
class FlightQuery:
    """Informations structurées de la requête de vol."""
    origin: str
    origin_code: str
    destination: str
    destination_code: str
    departure_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    infants: int = 0
    trip_type: str = "OW"  # OW (aller simple), RT (aller-retour), MC (multi-villes)
    cabin_class: str = "economy"
    flexible_dates: bool = False
    direct_only: bool = False
    raw_query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return asdict(self)
    
    def total_passengers(self) -> int:
        """Obtient le nombre total de passagers."""
        return self.adults + self.children + self.infants


class NLParser:
    """
    Analyseur de langage naturel pour les requêtes de vol.
    Utilise le LLM pour l'extraction d'entités et des utilitaires locaux pour la validation.
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialise l'analyseur.
        
        Args:
            use_llm: Si True, utilise le LLM pour l'extraction. Si False, utilise uniquement les expressions régulières.
        """
        self.use_llm = use_llm
        self.airport_db = AirportDatabase()
        self.date_parser = DateParser()
        self.ollama_url = f"{OLLAMA_CONFIG['base_url']}/api/generate"
        logger.info(f"NLParser initialisé avec use_llm={use_llm}, ollama_url={self.ollama_url}")
    
    def parse(self, query: str) -> Optional[FlightQuery]:
        """
        Analyse une requête de vol en langage naturel.
        
        Args:
            query: Requête en langage naturel (par exemple, "vol Paris New York 15 mars 2 adultes")
            
        Returns:
            Objet FlightQuery ou None si l'analyse échoue
        """
        logger.info(f"Parsing de la requête: '{query}'")
        
        if self.use_llm:
            logger.debug("Utilisation du LLM pour l'extraction")
            entities = self._extract_with_llm(query)
        else:
            logger.debug("Utilisation du regex pour l'extraction")
            entities = self._extract_with_regex(query)
        
        if not entities:
            logger.warning(f"Aucune entité extraite pour: '{query}'")
            return None
        
        logger.debug(f"Entités extraites: {entities}")

        # Complète les entités avec le regex si le LLM a loupé une variante (ex: "2 adulte", "a/r", "direct")
        if self.use_llm:
            logger.debug("Complément avec extraction regex")
            regex_entities = self._extract_with_regex(query)
            entities = self._merge_with_regex_fallback(entities, regex_entities)
            logger.debug(f"Entités après fusion: {entities}")
        
        result = self._build_flight_query(query, entities)
        if result:
            logger.info(f"FlightQuery créé: {result.origin_code} -> {result.destination_code}, {result.departure_date}")
        else:
            logger.warning(f"Échec de création du FlightQuery pour: '{query}'")
        return result
    
    def _extract_with_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """Extrait les entités en utilisant le LLM."""
        logger.debug(f"Extraction LLM pour: '{query}'")
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            query=query,
            current_date=datetime.now().strftime('%Y-%m-%d')
        )
        logger.debug(f"Prompt LLM: {prompt[:100]}...")
        
        try:
            logger.debug(f"Envoi de la requête à {self.ollama_url} avec le modèle {MODEL_NAME}")
            response = requests.post(
                self.ollama_url,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": OLLAMA_CONFIG["temperature"],
                        "num_predict": OLLAMA_CONFIG["num_predict"],
                    }
                },
                timeout=OLLAMA_CONFIG["timeout"]
            )
            response.raise_for_status()
            logger.debug(f"Réponse HTTP reçue avec statut: {response.status_code}")
            
            result = response.json()
            llm_response = result.get("response", "")
            logger.debug(f"Réponse LLM brute: {llm_response[:200]}..." if len(llm_response) > 200 else f"Réponse LLM brute: {llm_response}")
            
            # Extrait le JSON de la réponse
            json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                logger.debug(f"JSON extrait avec succès: {parsed}")
                return parsed
            
            # Tente d'analyser l'intégralité de la réponse comme JSON
            parsed = json.loads(llm_response)
            logger.debug(f"JSON parsé directement: {parsed}")
            return parsed
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"L'extraction LLM a échoué: {e}")
            logger.info("Fallback vers l'extraction regex")
            # Revient à l'extraction par expression régulière
            return self._extract_with_regex(query)
    
    def _extract_with_regex(self, query: str) -> Dict[str, Any]:
        """Extrait les entités en utilisant des motifs d'expressions régulières."""
        logger.debug(f"Extraction regex pour: '{query}'")
        entities = {
            "origin": None,
            "destination": None,
            "date": None,
            "return_date": None,
            "adults": 1,
            "children": 0,
            "infants": 0,
            "trip_type": "OW",
            "cabin_class": "economy",
            "flexible_dates": False,
            "direct_only": False,
        }
        
        query_lower = query.lower()
        
        # Extrait le nombre de passagers
        adult_match = re.search(r'(\d+)\s*(?:adulte|adult|pax|passager|personne)s?', query_lower)
        if adult_match:
            entities["adults"] = int(adult_match.group(1))
            logger.debug(f"Passagers adultes détectés: {entities['adults']}")
        
        child_match = re.search(r'(\d+)\s*(?:enfant|child|children|kid)s?', query_lower)
        if child_match:
            entities["children"] = int(child_match.group(1))
            logger.debug(f"Passagers enfants détectés: {entities['children']}")
        
        infant_match = re.search(r'(\d+)\s*(?:bébé|bebe|infant|baby|babies)', query_lower)
        if infant_match:
            entities["infants"] = int(infant_match.group(1))
            logger.debug(f"Passagers bébés détectés: {entities['infants']}")
        
        # Détecte le type de voyage
        if any(x in query_lower for x in ['aller-retour', 'aller retour', 'round trip', 'a/r', 'ar']):
            entities["trip_type"] = "RT"
            logger.debug("Type de voyage détecté: RT (aller-retour)")
        
        # Détecte la classe de cabine
        if any(x in query_lower for x in ['business', 'affaires', 'classe affaires']):
            entities["cabin_class"] = "business"
            logger.debug("Classe cabine détectée: business")
        elif any(x in query_lower for x in ['first', 'première', 'premiere']):
            entities["cabin_class"] = "first"
            logger.debug("Classe cabine détectée: first")
        elif any(x in query_lower for x in ['premium', 'economy+']):
            entities["cabin_class"] = "premium_economy"
            logger.debug("Classe cabine détectée: premium_economy")
        
        # Détecte les vols directs uniquement
        if any(x in query_lower for x in ['direct', 'sans escale', 'non-stop', 'nonstop']):
            entities["direct_only"] = True
            logger.debug("Vol direct uniquement détecté")
        
        # Détecte les dates flexibles
        if any(x in query_lower for x in ['flexible', 'environ', 'autour de', 'around']):
            entities["flexible_dates"] = True
            logger.debug("Dates flexibles détectées")
        
        # Extrait les villes - recherche des noms de villes dans la requête
        # Ceci est une extraction simplifiée - le LLM fait mieux
        # Gère les noms de villes composés comme "New York", "Los Angeles", etc.
        
        # Mots à exclure de la correspondance des villes (dates, mots courants, etc.)
        excluded_words = {
            'vol', 'pour', 'avec', 'dans', 'demain', 'après-demain', 'apres-demain',
            'hier', 'aujourdhui', "aujourd'hui", 'jour', 'jours', 'semaine', 'semaines',
            'mois', 'prochain', 'prochaine', 'lundi', 'mardi', 'mercredi', 'jeudi',
            'vendredi', 'samedi', 'dimanche', 'janvier', 'février', 'mars', 'avril',
            'mai', 'juin', 'juillet', 'août', 'aout', 'septembre', 'octobre',
            'novembre', 'décembre', 'decembre', 'adulte', 'adultes', 'enfant',
            'enfants', 'bébé', 'bebe', 'bébés', 'bebes', 'passager', 'passagers',
            'personne', 'personnes', 'aller', 'retour', 'aller-retour', 'business',
            'economy', 'économique', 'economique', 'première', 'premiere', 'first',
            'direct', 'sans', 'escale', 'flexible', 'environ', 'autour', 'classe',
            'billet', 'billets', 'cherche', 'recherche', 'veux', 'voudrais',
        }
        
        words = query.split()
        potential_cities = []  # Stockera des tuples (position, nom_ville)
        used_indices = set()
        
        # Premier passage : tente de faire correspondre des noms de villes de 2 mots
        for i in range(len(words) - 1):
            if i in used_indices:
                continue
            two_words = f"{words[i]} {words[i+1]}"
            two_words_lower = two_words.lower()
            # Passe si un mot est dans la liste des mots exclus
            if words[i].lower() in excluded_words or words[i+1].lower() in excluded_words:
                continue
            if len(two_words) >= 3:
                lookup = self.airport_db.lookup(two_words)
                if lookup:
                    code, info = lookup
                    # Pour les recherches de 2 mots, assurez-vous d'une correspondance exacte (pas floue)
                    is_good_match = (
                        two_words_lower == info.city.lower() or
                        two_words_lower in [a.lower() for a in info.aliases]
                    )
                    if is_good_match:
                        potential_cities.append((i, info.city))  # Stocke avec la position
                        used_indices.add(i)
                        used_indices.add(i + 1)
        
        # Deuxième passage : tente des mots uniques non encore utilisés
        for i, word in enumerate(words):
            if i in used_indices:
                continue
            word_lower = word.lower()
            # Passe les mots exclus
            if word_lower in excluded_words:
                continue
            # Vérifie les mots de 3 caractères ou plus, quelle que soit la casse - airport_db gère la normalisation
            if len(word) >= 3:
                lookup = self.airport_db.lookup(word)
                if lookup:
                    # Pour les mots uniques, assurez-vous d'une bonne correspondance (pas de "mauvaise" correspondance floue)
                    # Vérifie si le mot correspond réellement à la ville/alias de près
                    code, info = lookup
                    is_good_match = (
                        word_lower == info.city.lower() or
                        word_lower == code.lower() or
                        word_lower in [a.lower() for a in info.aliases]
                    )
                    if is_good_match:
                        potential_cities.append((i, info.city))  # Stocke avec la position
                        used_indices.add(i)
        
        # Trie par position dans la requête pour préserver l'ordre
        potential_cities.sort(key=lambda x: x[0])
        city_names = [city for _, city in potential_cities]
        
        if len(city_names) >= 2:
            entities["origin"] = city_names[0]
            entities["destination"] = city_names[1]
            logger.debug(f"Villes détectées - Origine: {entities['origin']}, Destination: {entities['destination']}")
        elif len(city_names) == 1:
            entities["destination"] = city_names[0]
            logger.debug(f"Une seule ville détectée comme destination: {entities['destination']}")
        
        # Extrait la date - laisse DateParser s'en charger
        # Motifs courants pour isoler la partie date (restreints aux noms de mois pour éviter les faux positifs comme "2 adulte")
        month_names = (
            "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|"
            "octobre|novembre|décembre|decembre|january|february|march|april|may|june|"
            "july|august|september|october|november|december"
        )
        date_patterns = [
            rf'(?:le\s+)?(\d{{1,2}}\s+(?:{month_names})(?:\s+\d{{4}})?)',  # le 15 janvier 2025
            r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',  # 15/01/2025
            r'(\d{4}-\d{2}-\d{2})',  # 2025-01-15
            r'(demain|aujourd\'?hui|après\-?demain)',  # relative
            r'(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s*(?:prochain)?',  # jour de la semaine
            r'dans\s+\d+\s+(?:jour|semaine|mois)s?',  # dans 2 semaines
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query_lower)
            if match:
                parsed_date = self.date_parser.parse(match.group(0))
                if parsed_date:
                    entities["date"] = parsed_date
                    logger.debug(f"Date détectée: {parsed_date} (expression: '{match.group(0)}')")
                    break
        
        logger.debug(f"Entités regex finales: {entities}")
        return entities

    def _merge_with_regex_fallback(self, primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complète les entités LLM avec les valeurs regex quand le LLM a laissé les valeurs par défaut.
        Couvre: passagers, trip_type, cabin_class, direct_only, flexible_dates.
        """
        logger.debug(f"Fusion des entités - LLM: {primary}, Regex: {fallback}")
        merged = dict(primary)
        
        # Valeurs par défaut pour chaque champ
        defaults = {
            # Compteurs passagers
            "adults": 1,
            "children": 0,
            "infants": 0,
            # Type de voyage et options
            "trip_type": "OW",
            "cabin_class": "economy",
            "direct_only": False,
            "flexible_dates": False,
        }

        for key, default_value in defaults.items():
            p_val = primary.get(key)
            f_val = fallback.get(key)

            # Si le LLM n'a rien mis ou a laissé la valeur par défaut, on préfère la valeur regex si elle est plus précise
            if (p_val is None or p_val == default_value) and f_val not in (None, default_value):
                merged[key] = f_val
                logger.debug(f"Remplacement de '{key}': {p_val} -> {f_val} (fallback regex)")

        logger.debug(f"Entités fusionnées: {merged}")
        return merged
    
    def _build_flight_query(self, raw_query: str, entities: Dict[str, Any]) -> Optional[FlightQuery]:
        """Construit un FlightQuery à partir des entités extraites."""
        logger.debug(f"Construction du FlightQuery à partir des entités: {entities}")
        # Résout les codes d'aéroport
        origin = entities.get("origin")
        destination = entities.get("destination")
        
        if not origin or not destination:
            logger.warning(f"Origine ou destination manquante - origin: {origin}, destination: {destination}")
            return None
        
        origin_code = self.airport_db.get_code(origin)
        destination_code = self.airport_db.get_code(destination)
        logger.debug(f"Codes aéroport résolus - {origin} -> {origin_code}, {destination} -> {destination_code}")
        
        if not origin_code or not destination_code:
            logger.warning(f"Impossible de résoudre les codes aéroport - origin_code: {origin_code}, destination_code: {destination_code}")
            return None
        
        # Analyse la date si elle n'est pas déjà au format ISO
        departure_date = entities.get("date")
        if departure_date:
            # Vérifie si déjà au format ISO
            if not re.match(r'\d{4}-\d{2}-\d{2}', str(departure_date)):
                departure_date = self.date_parser.parse(str(departure_date))
        else:
            # Par défaut, demain si aucune date n'est spécifiée
            departure_date = self.date_parser.parse("demain")
        
        return_date = entities.get("return_date")
        if return_date and not re.match(r'\d{4}-\d{2}-\d{2}', str(return_date)):
            return_date = self.date_parser.parse(str(return_date))
        
        return FlightQuery(
            origin=origin,
            origin_code=origin_code,
            destination=destination,
            destination_code=destination_code,
            departure_date=departure_date or "",
            return_date=return_date,
            adults=int(entities.get("adults", 1)),
            children=int(entities.get("children", 0)),
            infants=int(entities.get("infants", 0)),
            trip_type=entities.get("trip_type", "OW"),
            cabin_class=entities.get("cabin_class", "economy"),
            flexible_dates=bool(entities.get("flexible_dates", False)),
            direct_only=bool(entities.get("direct_only", False)),
            raw_query=raw_query
        )


if __name__ == "__main__":
    import sys
    
    # Teste l'analyseur
    parser = NLParser(use_llm=True)
    
    # Utilise l'argument de ligne de commande ou les requêtes de test par défaut
    if len(sys.argv) > 1:
        test_queries = [" ".join(sys.argv[1:])]
    else:
        test_queries = [
            "vol Paris Londres demain 2 adultes",
            "Je cherche un aller-retour Nice Rome du 15 au 22 mars",
            "billet New York 25/12/2025 1 adulte 1 enfant business",
            "vol direct Marseille Berlin dans 2 semaines",
        ]
    
    print("Test de l'analyseur NL")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nRequête : '{query}'")
        result = parser.parse(query)
        if result:
            print(f"  Origine : {result.origin} ({result.origin_code})")
            print(f"  Destination : {result.destination} ({result.destination_code})")
            print(f"  Date : {result.departure_date}")
            print(f"  Passagers : {result.adults}A + {result.children}C + {result.infants}I")
            print(f"  Type : {result.trip_type}")
            print(f"  Classe : {result.cabin_class}")
        else:
            print("  ❌ Analyse échouée")
