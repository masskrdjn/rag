# xft_builder.py
"""
Générateur XML XFT pour les requêtes de disponibilité de vol.
Génère un XML XFT valide à partir de données de requête de vol structurées.
"""
import os
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from .config_searchnl import XFT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "xft_builder.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TravellerInfo:
    """Informations sur un voyageur."""
    id: str
    type: str  # Adulte, Enfant, Bébé
    

class XFTBuilder:
    """
    Générateur pour les requêtes XML XFT de disponibilité de vol.
    """
    
    # Mappage des types de voyageurs
    TRAVELLER_TYPES = {
        "adult": "Adult",
        "child": "Child", 
        "infant": "Infant",
    }
    
    # Type de segment basé sur le type de voyage
    SEGMENT_TYPES = {
        "OW": "OneWay",
        "RT": "RoundTrip",
        "MC": "MultiCity",
    }
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialise le générateur XFT.
        
        Args:
            templates_dir: Chemin vers les fichiers de modèle XFT
        """
        self.templates_dir = templates_dir or XFT_CONFIG["templates_dir"]
        logger.info(f"XFTBuilder initialisé avec templates_dir={self.templates_dir}")
    
    def _load_template(self, trip_type: str = "OW") -> Optional[str]:
        """Charge un fichier de modèle XFT."""
        logger.debug(f"Chargement du template pour trip_type={trip_type}")
        template_dirs = {
            "OW": os.path.join(self.templates_dir, "flight", "OW"),
            "RT": os.path.join(self.templates_dir, "flight", "RT"),
            "MC": os.path.join(self.templates_dir, "flight", "MC"),
        }
        
        template_dir = template_dirs.get(trip_type, template_dirs["OW"])
        
        # Utilise le modèle d'aéroport par défaut
        template_file = os.path.join(template_dir, "search_metropolitan.xml")
        logger.debug(f"Chemin du template: {template_file}")
        
        if os.path.exists(template_file):
            with open(template_file, 'r', encoding='utf-8') as f:
                logger.debug(f"Template chargé avec succès: {template_file}")
                return f.read()
        
        logger.warning(f"Template non trouvé: {template_file}")
        return None
    
    def build(self, 
              origin_code: str,
              destination_code: str,
              departure_date: str,
              return_date: Optional[str] = None,
              adults: int = 1,
              children: int = 0,
              infants: int = 0,
              trip_type: str = "OW",
              **kwargs) -> str:
        """
        Construit une requête XML XFT.
        
        Args:
            origin_code: Code IATA de l'origine
            destination_code: Code IATA de la destination
            departure_date: Date de départ au format AAAA-MM-JJ
            return_date: Date de retour pour les voyages RT
            adults: Nombre de passagers adultes
            children: Nombre de passagers enfants (2-11 ans)
            infants: Nombre de passagers bébés (<2 ans)
            trip_type: OW, RT, ou MC
            
        Returns:
            Chaîne XML XFT
        """
        logger.info(f"Construction XFT: {origin_code} -> {destination_code}, date={departure_date}, type={trip_type}")
        logger.debug(f"Passagers: {adults}A + {children}C + {infants}I")
        
        # Crée l'élément racine
        root = ET.Element("Data")
        logger.debug("Élément racine 'Data' créé")
        
        # Ajoute l'espace de noms XML pour xsi
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        
        # Élément Action
        action = ET.SubElement(root, "Action")
        action.set("Purpose", "Get")
        action.set("Code", "Availability")
        logger.debug("Élément 'Action' ajouté")
        
        # Élément Trip
        trip = ET.SubElement(root, "Trip")
        logger.debug("Élément 'Trip' ajouté")
        
        # Construit le(s) segment(s)
        self._add_segment(trip, trip_type, adults + children)
        logger.debug("Segment ajouté")
        
        # Ajoute les voyageurs
        self._add_travellers(trip, adults, children, infants)
        logger.debug("Voyageurs ajoutés")
        
        # Ajoute les dates et les lieux
        self._add_itinerary(trip, origin_code, destination_code, 
                           departure_date, return_date, trip_type)
        logger.debug("Itinéraire ajouté")
        
        # Convertit en chaîne avec un formatage joli
        xml_output = self._pretty_print(root)
        logger.info(f"XML XFT généré avec succès ({len(xml_output)} caractères)")
        return xml_output
    
    def _add_segment(self, trip: ET.Element, trip_type: str, seat_count: int):
        """Ajoute les informations de segment au voyage."""
        logger.debug(f"Ajout segment: type={trip_type}, seats={seat_count}")
        segment_product = ET.SubElement(trip, "Segment")
        segment_product.set("xsi:type", "SegmentProductType")
        
        segments_wrapper = ET.SubElement(segment_product, "Segments")
        segments_wrapper.set("What", "Details")
        
        segment_air = ET.SubElement(segments_wrapper, "Segment")
        segment_air.set("xsi:type", "SegmentAirType")
        segment_air.set("Type", self.SEGMENT_TYPES.get(trip_type, "OneWay"))
        
        seat = ET.SubElement(segment_air, "Seat")
        seat.set("Quantity", str(seat_count))
    
    def _add_travellers(self, trip: ET.Element, adults: int, children: int, infants: int):
        """Ajoute les informations sur les voyageurs au voyage."""
        logger.debug(f"Ajout voyageurs: {adults}A + {children}C + {infants}I")
        travellers = ET.SubElement(trip, "Travellers")
        
        traveller_id = 1
        
        # Ajoute les adultes
        for _ in range(adults):
            traveller = ET.SubElement(travellers, "Traveller")
            traveller.set("ID", f"T{traveller_id}")
            traveller.set("Type", "Adult")
            traveller_id += 1
        logger.debug(f"{adults} adulte(s) ajouté(s)")
        
        # Ajoute les enfants
        for _ in range(children):
            traveller = ET.SubElement(travellers, "Traveller")
            traveller.set("ID", f"T{traveller_id}")
            traveller.set("Type", "Child")
            traveller_id += 1
        if children > 0:
            logger.debug(f"{children} enfant(s) ajouté(s)")
        
        # Ajoute les bébés
        for _ in range(infants):
            traveller = ET.SubElement(travellers, "Traveller")
            traveller.set("ID", f"T{traveller_id}")
            traveller.set("Type", "Infant")
            traveller_id += 1
        if infants > 0:
            logger.debug(f"{infants} bébé(s) ajouté(s)")
    
    def _add_itinerary(self, trip: ET.Element, origin: str, destination: str,
                       departure_date: str, return_date: Optional[str], trip_type: str):
        """Ajoute l'itinéraire (dates et lieux) au voyage."""
        logger.debug(f"Ajout itinéraire: {origin} -> {destination}, départ={departure_date}, retour={return_date}")
        
        # Date de départ
        begin = ET.SubElement(trip, "Begin")
        begin.set("Value", departure_date)
        logger.debug(f"Date de départ ajoutée: {departure_date}")
        
        # Origine
        from_elem = ET.SubElement(trip, "From")
        from_elem.set("xsi:type", "CityType")
        from_elem.set("Code", origin)
        logger.debug(f"Origine ajoutée: {origin}")
        
        # Destination
        to_elem = ET.SubElement(trip, "To")
        to_elem.set("xsi:type", "CityType")
        to_elem.set("Code", destination)
        logger.debug(f"Destination ajoutée: {destination}")
        
        # Date de retour pour les allers-retours
        if trip_type == "RT" and return_date:
            end = ET.SubElement(trip, "End")
            end.set("Value", return_date)
            logger.debug(f"Date de retour ajoutée: {return_date}")
    
    def _pretty_print(self, elem: ET.Element) -> str:
        """Affiche joliment le XML avec une indentation correcte."""
        logger.debug("Formatage du XML")
        rough_string = ET.tostring(elem, encoding='unicode')
        
        # Ajoute la déclaration XML
        rough_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + rough_string
        
        # Utilise minidom pour un affichage joli
        dom = minidom.parseString(rough_string)
        pretty = dom.toprettyxml(indent="    ")
        
        # Supprime les lignes vides supplémentaires et la déclaration en double
        lines = pretty.split('\n')
        result_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            # Saute la première ligne de déclaration (minidom ajoute la sienne)
            if i == 0 and '<?xml' in line:
                continue
            # Saute les lignes vides après les éléments
            if line.strip() == '' and skip_next_empty:
                skip_next_empty = False
                continue
            if line.strip():
                result_lines.append(line)
                skip_next_empty = False
            else:
                skip_next_empty = True
        
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + '\n'.join(result_lines)
    
    def build_from_query(self, flight_query) -> str:
        """
        Construit un XML XFT à partir d'un objet FlightQuery.
        
        Args:
            flight_query: Instance FlightQuery de nl_parser
            
        Returns:
            Chaîne XML XFT
        """
        logger.info(f"Construction XFT depuis FlightQuery: {flight_query.origin_code} -> {flight_query.destination_code}")
        return self.build(
            origin_code=flight_query.origin_code,
            destination_code=flight_query.destination_code,
            departure_date=flight_query.departure_date,
            return_date=flight_query.return_date,
            adults=flight_query.adults,
            children=flight_query.children,
            infants=flight_query.infants,
            trip_type=flight_query.trip_type,
        )
    
    def validate(self, xml_string: str) -> bool:
        """
        Valide la structure XML XFT.
        
        Args:
            xml_string: XML XFT à valider
            
        Returns:
            True si valide, False sinon
        """
        logger.debug("Validation du XML XFT")
        try:
            root = ET.fromstring(xml_string)
            
            # Vérifie les éléments requis
            action = root.find("Action")
            if action is None:
                logger.warning("Validation échouée: élément 'Action' manquant")
                return False
            
            trip = root.find("Trip")
            if trip is None:
                logger.warning("Validation échouée: élément 'Trip' manquant")
                return False
            
            # Vérifie les éléments de voyage requis
            required = ["Segment", "Travellers", "Begin", "From", "To"]
            for elem_name in required:
                if trip.find(elem_name) is None:
                    logger.warning(f"Validation échouée: élément '{elem_name}' manquant dans Trip")
                    return False
            
            logger.debug("Validation XML réussie")
            return True
            
        except ET.ParseError as e:
            logger.error(f"Erreur de parsing XML: {e}")
            return False


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Génère du XML XFT")
    parser.add_argument("--origin", "-o", required=True, help="Code IATA de l'origine")
    parser.add_argument("--dest", "-d", required=True, help="Code IATA de la destination")
    parser.add_argument("--date", required=True, help="Date de départ (AAAA-MM-JJ)")
    parser.add_argument("--return-date", "-r", help="Date de retour pour les RT")
    parser.add_argument("--adults", "-a", type=int, default=1, help="Nombre d'adultes")
    parser.add_argument("--children", "-c", type=int, default=0, help="Nombre d'enfants")
    parser.add_argument("--infants", "-i", type=int, default=0, help="Nombre de bébés")
    parser.add_argument("--type", "-t", choices=["OW", "RT", "MC"], default="OW")
    
    args = parser.parse_args()
    
    builder = XFTBuilder()
    xml_output = builder.build(
        origin_code=args.origin,
        destination_code=args.dest,
        departure_date=args.date,
        return_date=args.return_date,
        adults=args.adults,
        children=args.children,
        infants=args.infants,
        trip_type=args.type,
    )
    
    print(xml_output)
    
    # Valide
    if builder.validate(xml_output):
        print("\n✓ Le XML est valide", file=sys.stderr)
    else:
        print("\n✗ La validation XML a échoué", file=sys.stderr)
