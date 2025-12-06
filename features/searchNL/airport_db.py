# airport_db.py
"""
Base de données des codes d'aéroport et de ville avec support de correspondance floue.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "airport_db.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AirportInfo:
    """Informations sur un aéroport ou une ville."""
    code: str
    name: str
    city: str
    country: str
    is_metropolitan: bool = False
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


# =============================================================================
# BASE DE DONNÉES AÉROPORT/VILLE
# =============================================================================

# Principaux aéroports et zones métropolitaines pour l'Europe et les destinations courantes
AIRPORTS_DB: Dict[str, AirportInfo] = {
    # France - Paris
    "PAR": AirportInfo("PAR", "Paris (Tous Aéroports)", "Paris", "FR", True, 
                       ["paris", "paname"]),
    "CDG": AirportInfo("CDG", "Charles de Gaulle", "Paris", "FR", False,
                       ["roissy", "charles de gaulle", "cdg"]),
    "ORY": AirportInfo("ORY", "Paris Orly", "Paris", "FR", False,
                       ["orly"]),
    "BVA": AirportInfo("BVA", "Paris Beauvais", "Paris", "FR", False,
                       ["beauvais"]),
    
    # France - Autres villes
    "NCE": AirportInfo("NCE", "Nice Côte d'Azur", "Nice", "FR", False,
                       ["nice", "cote d'azur"]),
    "MRS": AirportInfo("MRS", "Marseille Provence", "Marseille", "FR", False,
                       ["marseille", "marignane"]),
    "LYS": AirportInfo("LYS", "Lyon Saint-Exupéry", "Lyon", "FR", False,
                       ["lyon", "saint exupery", "satolas"]),
    "TLS": AirportInfo("TLS", "Toulouse Blagnac", "Toulouse", "FR", False,
                       ["toulouse", "blagnac"]),
    "BOD": AirportInfo("BOD", "Bordeaux Mérignac", "Bordeaux", "FR", False,
                       ["bordeaux", "merignac"]),
    "NTE": AirportInfo("NTE", "Nantes Atlantique", "Nantes", "FR", False,
                       ["nantes"]),
    "SXB": AirportInfo("SXB", "Strasbourg", "Strasbourg", "FR", False,
                       ["strasbourg"]),
    "LIL": AirportInfo("LIL", "Lille Lesquin", "Lille", "FR", False,
                       ["lille", "lesquin"]),
    
    # Royaume-Uni - Londres
    "LON": AirportInfo("LON", "Londres (Tous Aéroports)", "Londres", "GB", True,
                       ["londres", "london"]),
    "LHR": AirportInfo("LHR", "Londres Heathrow", "Londres", "GB", False,
                       ["heathrow"]),
    "LGW": AirportInfo("LGW", "Londres Gatwick", "Londres", "GB", False,
                       ["gatwick"]),
    "STN": AirportInfo("STN", "Londres Stansted", "Londres", "GB", False,
                       ["stansted"]),
    "LTN": AirportInfo("LTN", "Londres Luton", "Londres", "GB", False,
                       ["luton"]),
    "LCY": AirportInfo("LCY", "Londres City", "Londres", "GB", False,
                       ["city airport"]),
    
    # Royaume-Uni - Autres
    "MAN": AirportInfo("MAN", "Manchester", "Manchester", "GB", False,
                       ["manchester"]),
    "EDI": AirportInfo("EDI", "Édimbourg", "Édimbourg", "GB", False,
                       ["edinburgh", "edimbourg"]),
    "BHX": AirportInfo("BHX", "Birmingham", "Birmingham", "GB", False,
                       ["birmingham"]),
    
    # Allemagne
    "BER": AirportInfo("BER", "Berlin Brandenburg", "Berlin", "DE", False,
                       ["berlin"]),
    "FRA": AirportInfo("FRA", "Francfort", "Francfort", "DE", False,
                       ["francfort", "frankfurt"]),
    "MUC": AirportInfo("MUC", "Munich", "Munich", "DE", False,
                       ["munich", "munchen", "münchen"]),
    "DUS": AirportInfo("DUS", "Düsseldorf", "Düsseldorf", "DE", False,
                       ["dusseldorf", "düsseldorf"]),
    "HAM": AirportInfo("HAM", "Hambourg", "Hambourg", "DE", False,
                       ["hamburg", "hambourg"]),
    
    # Espagne
    "MAD": AirportInfo("MAD", "Madrid Barajas", "Madrid", "ES", False,
                       ["madrid", "barajas"]),
    "BCN": AirportInfo("BCN", "Barcelone El Prat", "Barcelone", "ES", False,
                       ["barcelone", "barcelona", "el prat"]),
    "AGP": AirportInfo("AGP", "Malaga", "Malaga", "ES", False,
                       ["malaga"]),
    "PMI": AirportInfo("PMI", "Palma de Majorque", "Palma", "ES", False,
                       ["palma", "majorque", "mallorca"]),
    "IBZ": AirportInfo("IBZ", "Ibiza", "Ibiza", "ES", False,
                       ["ibiza"]),
    
    # Italie
    "ROM": AirportInfo("ROM", "Rome (Tous Aéroports)", "Rome", "IT", True,
                       ["rome", "roma"]),
    "FCO": AirportInfo("FCO", "Rome Fiumicino", "Rome", "IT", False,
                       ["fiumicino", "leonardo da vinci"]),
    "CIA": AirportInfo("CIA", "Rome Ciampino", "Rome", "IT", False,
                       ["ciampino"]),
    "MIL": AirportInfo("MIL", "Milan (Tous Aéroports)", "Milan", "IT", True,
                       ["milan", "milano"]),
    "MXP": AirportInfo("MXP", "Milan Malpensa", "Milan", "IT", False,
                       ["malpensa"]),
    "LIN": AirportInfo("LIN", "Milan Linate", "Milan", "IT", False,
                       ["linate"]),
    "VCE": AirportInfo("VCE", "Venise Marco Polo", "Venise", "IT", False,
                       ["venise", "venice", "venezia"]),
    "NAP": AirportInfo("NAP", "Naples", "Naples", "IT", False,
                       ["naples", "napoli"]),
    "FLR": AirportInfo("FLR", "Florence", "Florence", "IT", False,
                       ["florence", "firenze"]),
    
    # Pays-Bas
    "AMS": AirportInfo("AMS", "Amsterdam Schiphol", "Amsterdam", "NL", False,
                       ["amsterdam", "schiphol"]),
    
    # Belgique
    "BRU": AirportInfo("BRU", "Bruxelles", "Bruxelles", "BE", False,
                       ["bruxelles", "brussels"]),
    "CRL": AirportInfo("CRL", "Bruxelles Sud Charleroi", "Charleroi", "BE", False,
                       ["charleroi"]),
    
    # Suisse
    "GVA": AirportInfo("GVA", "Genève", "Genève", "CH", False,
                       ["geneve", "geneva", "genf"]),
    "ZRH": AirportInfo("ZRH", "Zurich", "Zurich", "CH", False,
                       ["zurich", "zürich"]),
    
    # Portugal
    "LIS": AirportInfo("LIS", "Lisbonne", "Lisbonne", "PT", False,
                       ["lisbonne", "lisbon", "lisboa"]),
    "OPO": AirportInfo("OPO", "Porto", "Porto", "PT", False,
                       ["porto"]),
    "FAO": AirportInfo("FAO", "Faro", "Faro", "PT", False,
                       ["faro", "algarve"]),
    
    # Grèce
    "ATH": AirportInfo("ATH", "Athènes", "Athènes", "GR", False,
                       ["athenes", "athens"]),
    "SKG": AirportInfo("SKG", "Thessalonique", "Thessalonique", "GR", False,
                       ["thessalonique", "thessaloniki"]),
    "HER": AirportInfo("HER", "Héraklion", "Crète", "GR", False,
                       ["heraklion", "crete"]),
    
    # Autres Européens
    "VIE": AirportInfo("VIE", "Vienne", "Vienne", "AT", False,
                       ["vienne", "vienna", "wien"]),
    "PRG": AirportInfo("PRG", "Prague", "Prague", "CZ", False,
                       ["prague", "praha"]),
    "BUD": AirportInfo("BUD", "Budapest", "Budapest", "HU", False,
                       ["budapest"]),
    "WAW": AirportInfo("WAW", "Varsovie", "Varsovie", "PL", False,
                       ["varsovie", "warsaw"]),
    "CPH": AirportInfo("CPH", "Copenhague", "Copenhague", "DK", False,
                       ["copenhague", "copenhagen"]),
    "OSL": AirportInfo("OSL", "Oslo", "Oslo", "NO", False,
                       ["oslo"]),
    "ARN": AirportInfo("ARN", "Stockholm Arlanda", "Stockholm", "SE", False,
                       ["stockholm"]),
    "HEL": AirportInfo("HEL", "Helsinki", "Helsinki", "FI", False,
                       ["helsinki"]),
    "DUB": AirportInfo("DUB", "Dublin", "Dublin", "IE", False,
                       ["dublin"]),
    
    # Moyen-Orient et Afrique
    "DXB": AirportInfo("DXB", "Dubaï", "Dubaï", "AE", False,
                       ["dubai", "dubaï"]),
    "DOH": AirportInfo("DOH", "Doha Hamad", "Doha", "QA", False,
                       ["doha"]),
    "IST": AirportInfo("IST", "Istanbul", "Istanbul", "TR", False,
                       ["istanbul"]),
    "CAI": AirportInfo("CAI", "Le Caire", "Le Caire", "EG", False,
                       ["caire", "cairo"]),
    "CMN": AirportInfo("CMN", "Casablanca", "Casablanca", "MA", False,
                       ["casablanca", "casa"]),
    "RAK": AirportInfo("RAK", "Marrakech", "Marrakech", "MA", False,
                       ["marrakech"]),
    "TUN": AirportInfo("TUN", "Tunis", "Tunis", "TN", False,
                       ["tunis"]),
    "ALG": AirportInfo("ALG", "Alger", "Alger", "DZ", False,
                       ["alger", "algiers"]),
    
    # Amériques
    "NYC": AirportInfo("NYC", "New York (Tous Aéroports)", "New York", "US", True,
                       ["new york", "ny"]),
    "JFK": AirportInfo("JFK", "New York JFK", "New York", "US", False,
                       ["jfk", "kennedy"]),
    "EWR": AirportInfo("EWR", "Newark", "New York", "US", False,
                       ["newark"]),
    "LGA": AirportInfo("LGA", "LaGuardia", "New York", "US", False,
                       ["laguardia"]),
    "LAX": AirportInfo("LAX", "Los Angeles", "Los Angeles", "US", False,
                       ["los angeles", "la"]),
    "SFO": AirportInfo("SFO", "San Francisco", "San Francisco", "US", False,
                       ["san francisco", "sf"]),
    "MIA": AirportInfo("MIA", "Miami", "Miami", "US", False,
                       ["miami"]),
    "ORD": AirportInfo("ORD", "Chicago O'Hare", "Chicago", "US", False,
                       ["chicago"]),
    "BOS": AirportInfo("BOS", "Boston", "Boston", "US", False,
                       ["boston"]),
    "WAS": AirportInfo("WAS", "Washington (Tous Aéroports)", "Washington", "US", True,
                       ["washington"]),
    "YUL": AirportInfo("YUL", "Montréal", "Montréal", "CA", False,
                       ["montreal", "montréal"]),
    "YYZ": AirportInfo("YYZ", "Toronto Pearson", "Toronto", "CA", False,
                       ["toronto"]),
    "MEX": AirportInfo("MEX", "Mexico City", "Mexico City", "MX", False,
                       ["mexico", "mexico city"]),
    
    # Asie
    "BKK": AirportInfo("BKK", "Bangkok", "Bangkok", "TH", False,
                       ["bangkok"]),
    "SIN": AirportInfo("SIN", "Singapour Changi", "Singapour", "SG", False,
                       ["singapour", "singapore"]),
    "HKG": AirportInfo("HKG", "Hong Kong", "Hong Kong", "HK", False,
                       ["hong kong"]),
    "TYO": AirportInfo("TYO", "Tokyo (Tous Aéroports)", "Tokyo", "JP", True,
                       ["tokyo"]),
    "NRT": AirportInfo("NRT", "Tokyo Narita", "Tokyo", "JP", False,
                       ["narita"]),
    "HND": AirportInfo("HND", "Tokyo Haneda", "Tokyo", "JP", False,
                       ["haneda"]),
    "PEK": AirportInfo("PEK", "Pékin Capital", "Pékin", "CN", False,
                       ["pekin", "beijing"]),
    "PVG": AirportInfo("PVG", "Shanghai Pudong", "Shanghai", "CN", False,
                       ["shanghai"]),
    "DEL": AirportInfo("DEL", "Delhi", "Delhi", "IN", False,
                       ["delhi", "new delhi"]),
    "BOM": AirportInfo("BOM", "Bombay", "Bombay", "IN", False,
                       ["mumbai", "bombay"]),
    
    # Océanie
    "SYD": AirportInfo("SYD", "Sydney", "Sydney", "AU", False,
                       ["sydney"]),
    "MEL": AirportInfo("MEL", "Melbourne", "Melbourne", "AU", False,
                       ["melbourne"]),
    "AKL": AirportInfo("AKL", "Auckland", "Auckland", "NZ", False,
                       ["auckland"]),
}


class AirportDatabase:
    """Base de données pour les recherches de codes d'aéroport et de ville avec correspondance floue."""
    
    def __init__(self, fuzzy_threshold: int = 80, prefer_metropolitan: bool = True):
        """
        Initialise la base de données des aéroports.
        
        Args:
            fuzzy_threshold: Score minimum (0-100) pour les correspondances floues
            prefer_metropolitan: Si True, préfère les codes de ville aux codes d'aéroport
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.prefer_metropolitan = prefer_metropolitan
        logger.info(f"AirportDatabase initialisé: fuzzy_threshold={fuzzy_threshold}, prefer_metropolitan={prefer_metropolitan}")
        self._build_search_index()
    
    def _build_search_index(self):
        """Construit un index pour une recherche rapide."""
        logger.debug("Construction de l'index de recherche...")
        self.search_index = {}
        # Premier passage : indexer les aéroports non métropolitains
        for code, info in AIRPORTS_DB.items():
            if info.is_metropolitan:
                continue  # Ignorer les codes métropolitains au premier passage
            # Indexer par code
            self.search_index[code.lower()] = code
            # Indexer par nom
            self.search_index[info.name.lower()] = code
            # Indexer par ville (peut être écrasé par un code métropolitain)
            self.search_index[info.city.lower()] = code
            # Indexer par alias
            for alias in info.aliases:
                self.search_index[alias.lower()] = code
        
        # Deuxième passage : indexer les codes métropolitains (ils ont priorité sur les recherches par ville)
        metro_count = 0
        for code, info in AIRPORTS_DB.items():
            if not info.is_metropolitan:
                continue
            metro_count += 1
            # Indexer par code
            self.search_index[code.lower()] = code
            # Indexer par nom
            self.search_index[info.name.lower()] = code
            # Indexer par ville - le code métropolitain a la priorité
            self.search_index[info.city.lower()] = code
            # Indexer par alias
            for alias in info.aliases:
                self.search_index[alias.lower()] = code
        
        logger.info(f"Index construit: {len(self.search_index)} entrées, {len(AIRPORTS_DB)} aéroports, {metro_count} métropolitains")
    
    def _normalize(self, text: str) -> str:
        """Normalise le texte pour la recherche."""
        # Supprimer les accents et convertir en minuscules
        import unicodedata
        original = text
        text = text.lower().strip()
        # Normaliser les caractères Unicode
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))
        logger.debug(f"Normalisation: '{original}' -> '{text}'")
        return text
    
    def lookup(self, query: str) -> Optional[Tuple[str, AirportInfo]]:
        """
        Recherche un aéroport par code, nom, ville ou alias.
        
        Args:
            query: Terme de recherche (code, nom de ville, alias, etc.)
            
        Returns:
            Tuple (code, AirportInfo) ou None si non trouvé
        """
        logger.debug(f"Lookup pour: '{query}'")
        normalized = self._normalize(query)
        
        # Recherche directe dans l'index
        if normalized in self.search_index:
            code = self.search_index[normalized]
            logger.debug(f"Correspondance directe trouvée: '{query}' -> {code}")
            return (code, AIRPORTS_DB[code])
        
        # Recherche floue si pas de correspondance directe
        logger.debug(f"Pas de correspondance directe, recherche floue pour: '{query}'")
        best_match = None
        best_score = 0
        
        for key, code in self.search_index.items():
            # Score de similarité simple basé sur le ratio de caractères communs
            score = self._similarity_score(normalized, key)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = code
        
        if best_match:
            logger.debug(f"Correspondance floue trouvée: '{query}' -> {best_match} (score: {best_score})")
            return (best_match, AIRPORTS_DB[best_match])
        
        logger.debug(f"Aucune correspondance trouvée pour: '{query}'")
        return None
    
    def _similarity_score(self, s1: str, s2: str) -> int:
        """Calcule un score de similarité entre deux chaînes (0-100)."""
        if not s1 or not s2:
            return 0
        
        # Si l'une est contenue dans l'autre
        if s1 in s2 or s2 in s1:
            return 90
        
        # Calcul du ratio de caractères communs (Jaccard-like)
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0
        
        return int((intersection / union) * 100)
    
    def get_code(self, city_or_airport: str) -> Optional[str]:
        """
        Obtient le code IATA pour une ville ou un aéroport.
        
        Args:
            city_or_airport: Nom de la ville ou de l'aéroport
            
        Returns:
            Code IATA ou None si non trouvé
        """
        logger.debug(f"get_code pour: '{city_or_airport}'")
        result = self.lookup(city_or_airport)
        if result:
            code, info = result
            # Si on préfère les codes métropolitains et qu'un existe pour cette ville
            if self.prefer_metropolitan:
                # Chercher si un code métropolitain existe pour cette ville
                for metro_code, metro_info in AIRPORTS_DB.items():
                    if metro_info.is_metropolitan and metro_info.city.lower() == info.city.lower():
                        logger.debug(f"Code métropolitain trouvé: {metro_code} pour {city_or_airport}")
                        return metro_code
            logger.debug(f"Code retourné: {code} pour {city_or_airport}")
            return code
        logger.warning(f"Aucun code trouvé pour: '{city_or_airport}'")
        return None
    
    def search(self, query: str, limit: int = 5) -> List[Tuple[str, AirportInfo, int]]:
        """
        Recherche des aéroports correspondant à une requête.
        
        Args:
            query: Terme de recherche
            limit: Nombre maximum de résultats
            
        Returns:
            Liste de tuples (code, AirportInfo, score)
        """
        logger.debug(f"Recherche aéroports: '{query}', limit={limit}")
        normalized = self._normalize(query)
        results = []
        
        for code, info in AIRPORTS_DB.items():
            # Calculer le score pour différents champs
            scores = [
                self._similarity_score(normalized, self._normalize(code)),
                self._similarity_score(normalized, self._normalize(info.name)),
                self._similarity_score(normalized, self._normalize(info.city)),
            ]
            # Ajouter les scores des alias
            for alias in info.aliases:
                scores.append(self._similarity_score(normalized, self._normalize(alias)))
            
            best_score = max(scores)
            if best_score >= self.fuzzy_threshold // 2:  # Seuil plus bas pour la recherche
                results.append((code, info, best_score))
        
        # Trier par score décroissant et limiter
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:limit]
        logger.info(f"Recherche '{query}': {len(results)} résultat(s) trouvé(s)")
        return results
