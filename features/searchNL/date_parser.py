import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import re
import locale

# Configure logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "date_parser.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DateParser:
    """Analyse les dates en langage naturel au format ISO."""
    
    # Noms des jours en français
    DAYS_FR = {
        'lundi': 0, 'mardi': 1, 'mercredi': 2, 'jeudi': 3,
        'vendredi': 4, 'samedi': 5, 'dimanche': 6
    }
    
    # Noms des mois en français
    MONTHS_FR = {
        'janvier': 1, 'février': 2, 'fevrier': 2, 'mars': 3,
        'avril': 4, 'mai': 5, 'juin': 6, 'juillet': 7,
        'août': 8, 'aout': 8, 'septembre': 9, 'octobre': 10,
        'novembre': 11, 'décembre': 12, 'decembre': 12
    }
    
    # Noms des mois en anglais
    MONTHS_EN = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    # Expressions de date relatives
    RELATIVE_FR = {
        "aujourd'hui": 0, "aujourdhui": 0, "aujourd hui": 0,
        'demain': 1,
        'après-demain': 2, 'apres-demain': 2, 'après demain': 2, "apres demain": 2,
    }
    
    RELATIVE_EN = {
        'today': 0,
        'tomorrow': 1,
        'day after tomorrow': 2,
    }
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialise l'analyseur de dates.
        
        Args:
            reference_date: Date de référence pour l'analyse relative (par défaut : aujourd'hui)
        """
        self.reference = reference_date or datetime.now()
        logger.debug(f"DateParser initialized with reference date: {self.reference.strftime('%Y-%m-%d')}")
    
    def parse(self, text: str) -> Optional[str]:
        """
        Analyse une expression de date en langage naturel.
        
        Args:
            text: Expression de date (par exemple, "15 janvier", "demain", "dans 2 semaines")
            
        Returns:
            Date au format ISO (AAAA-MM-JJ) ou None si l'analyse échoue
        """
        logger.info(f"Parsing input text: '{text}'")
        text = text.lower().strip()
        
        # Essayer différentes stratégies d'analyse
        result = None
        
        strategies = [
            (self._parse_relative, "relative"),
            (self._parse_french_date, "French date"),
            (self._parse_english_date, "English date"),
            (self._parse_numeric, "numeric"),
            (self._parse_duration, "duration"),
            (self._parse_weekday, "weekday")
        ]
        
        for parser_func, strategy_name in strategies:
            logger.debug(f"Attempting to parse '{text}' with {strategy_name} strategy.")
            result = parser_func(text)
            if result:
                logger.debug(f"Successfully parsed '{text}' as {strategy_name}: {result.strftime('%Y-%m-%d')}")
                break
            else:
                logger.debug(f"{strategy_name} strategy failed for '{text}'.")
        
        if result:
            final_result = result.strftime('%Y-%m-%d')
            logger.info(f"Parsing successful for '{text}', result: {final_result}")
            return final_result
        else:
            logger.warning(f"Failed to parse '{text}' using any strategy.")
            return None
    
    def _parse_relative(self, text: str) -> Optional[datetime]:
        """Analyse les expressions relatives comme 'demain', 'today'."""
        logger.debug(f"Entering _parse_relative for '{text}'")
        for expr, days in {**self.RELATIVE_FR, **self.RELATIVE_EN}.items():
            if expr in text:
                parsed_date = self.reference + timedelta(days=days)
                logger.debug(f"Found relative expression '{expr}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
        logger.debug(f"No relative expression found for '{text}'")
        return None
    
    def _parse_french_date(self, text: str) -> Optional[datetime]:
        """Analyse le format de date français comme '15 janvier 2025' ou 'le 15 janvier'."""
        logger.debug(f"Entering _parse_french_date for '{text}'")
        pattern = r'(?:le\s+)?(\d{1,2})\s+([a-zéûô]+)(?:\s+(\d{4}))?'
        match = re.search(pattern, text)
        
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            year = int(match.group(3)) if match.group(3) else None
            
            month = self.MONTHS_FR.get(month_name)
            if month:
                if year is None:
                    year = self.reference.year
                    candidate = datetime(year, month, day)
                    if candidate < self.reference:
                        year += 1
                
                try:
                    parsed_date = datetime(year, month, day)
                    logger.debug(f"Found French date '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                    return parsed_date
                except ValueError:
                    logger.debug(f"Invalid French date components: year={year}, month={month}, day={day}")
                    return None
        logger.debug(f"No French date pattern matched for '{text}'")
        return None
    
    def _parse_english_date(self, text: str) -> Optional[datetime]:
        """Analyse le format de date anglais comme 'January 15, 2025' ou '15th January'."""
        logger.debug(f"Entering _parse_english_date for '{text}'")
        
        # Motif : Mois JJ[th/st/nd/rd] [AAAA]
        pattern1 = r'([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?'
        match1 = re.search(pattern1, text)
        
        if match1:
            month_name = match1.group(1)
            day = int(match1.group(2))
            year = int(match1.group(3)) if match1.group(3) else None
            
            month = self.MONTHS_EN.get(month_name)
            if month:
                if year is None:
                    year = self.reference.year
                    candidate = datetime(year, month, day)
                    if candidate < self.reference:
                        year += 1
                
                try:
                    parsed_date = datetime(year, month, day)
                    logger.debug(f"Found English date (M D) '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                    return parsed_date
                except ValueError:
                    logger.debug(f"Invalid English date components (M D): year={year}, month={month}, day={day}")
                    return None
        
        # Motif : JJ[th] Mois [AAAA]
        pattern2 = r'(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:\s+(\d{4}))?'
        match2 = re.search(pattern2, text)
        
        if match2:
            day = int(match2.group(1))
            month_name = match2.group(2)
            year = int(match2.group(3)) if match2.group(3) else None
            
            month = self.MONTHS_EN.get(month_name)
            if month:
                if year is None:
                    year = self.reference.year
                    candidate = datetime(year, month, day)
                    if candidate < self.reference:
                        year += 1
                
                try:
                    parsed_date = datetime(year, month, day)
                    logger.debug(f"Found English date (D M) '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                    return parsed_date
                except ValueError:
                    logger.debug(f"Invalid English date components (D M): year={year}, month={month}, day={day}")
                    return None
        
        logger.debug(f"No English date pattern matched for '{text}'")
        return None
    
    def _parse_numeric(self, text: str) -> Optional[datetime]:
        """Analyse les formats numériques comme JJ/MM/AAAA ou AAAA-MM-JJ."""
        logger.debug(f"Entering _parse_numeric for '{text}'")
        # Format ISO : AAAA-MM-JJ
        iso_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', text)
        if iso_match:
            try:
                parsed_date = datetime(
                    int(iso_match.group(1)),
                    int(iso_match.group(2)),
                    int(iso_match.group(3))
                )
                logger.debug(f"Found ISO numeric date '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
            except ValueError:
                logger.debug(f"Invalid ISO numeric date: {iso_match.groups()}")
                pass
        
        # Format français : JJ/MM/AAAA ou JJ/MM/AA
        fr_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})', text)
        if fr_match:
            day = int(fr_match.group(1))
            month = int(fr_match.group(2))
            year = int(fr_match.group(3))
            if year < 100:
                year += 2000 # Supposer le 21e siècle
            
            try:
                parsed_date = datetime(year, month, day)
                logger.debug(f"Found French numeric date '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
            except ValueError:
                logger.debug(f"Invalid French numeric date: year={year}, month={month}, day={day}")
                pass
        
        logger.debug(f"No numeric date pattern matched for '{text}'")
        return None
    
    def _parse_duration(self, text: str) -> Optional[datetime]:
        """Analyse les expressions de durée comme 'dans 2 semaines', 'in 3 days'."""
        logger.debug(f"Entering _parse_duration for '{text}'")
        # Français : dans X jours/semaines/mois
        fr_match = re.search(r'dans\s+(\d+)\s+(jour|jours|semaine|semaines|mois)', text)
        if fr_match:
            num = int(fr_match.group(1))
            unit = fr_match.group(2)
            
            if 'jour' in unit:
                parsed_date = self.reference + timedelta(days=num)
                logger.debug(f"Found French duration '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
            elif 'semaine' in unit:
                parsed_date = self.reference + timedelta(weeks=num)
                logger.debug(f"Found French duration '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
            elif 'mois' in unit:
                # Approximer les mois à 30 jours
                parsed_date = self.reference + timedelta(days=num * 30)
                logger.debug(f"Found French duration '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
        
        # Anglais : in X days/weeks/months
        en_match = re.search(r'in\s+(\d+)\s+(day|days|week|weeks|month|months)', text)
        if en_match:
            num = int(en_match.group(1))
            unit = en_match.group(2)
            
            if 'day' in unit:
                parsed_date = self.reference + timedelta(days=num)
                logger.debug(f"Found English duration '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
            elif 'week' in unit:
                parsed_date = self.reference + timedelta(weeks=num)
                logger.debug(f"Found English duration '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
            elif 'month' in unit:
                parsed_date = self.reference + timedelta(days=num * 30)
                logger.debug(f"Found English duration '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
        
        logger.debug(f"No duration pattern matched for '{text}'")
        return None
    
    def _parse_weekday(self, text: str) -> Optional[datetime]:
        """Analyse les expressions de jour de la semaine comme 'lundi prochain', 'next Monday'."""
        logger.debug(f"Entering _parse_weekday for '{text}'")
        # Français : [prochain] lundi
        for day_name, day_num in self.DAYS_FR.items():
            if day_name in text:
                force_next = 'prochain' in text or 'prochaine' in text
                parsed_date = self._next_weekday(day_num, force_next)
                logger.debug(f"Found French weekday '{day_name}' in '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
        
        # Jours de la semaine en anglais
        days_en = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        for day_name, day_num in days_en.items():
            if day_name in text:
                force_next = 'next' in text
                parsed_date = self._next_weekday(day_num, force_next)
                logger.debug(f"Found English weekday '{day_name}' in '{text}', returning {parsed_date.strftime('%Y-%m-%d')}")
                return parsed_date
        
        logger.debug(f"No weekday pattern matched for '{text}'")
        return None
    
    def _next_weekday(self, target_weekday: int, force_next_week: bool = False) -> datetime:
        """Obtient la prochaine occurrence d'un jour de la semaine."""
        current_weekday = self.reference.weekday()
        days_ahead = target_weekday - current_weekday
        
        if days_ahead <= 0 or force_next_week:
            days_ahead += 7
        
        calculated_date = self.reference + timedelta(days=days_ahead)
        logger.debug(f"Calculated next weekday (target: {target_weekday}, current: {current_weekday}, days_ahead: {days_ahead}, force_next_week: {force_next_week}, result: {calculated_date.strftime('%Y-%m-%d')}")
        return calculated_date
    
    def parse_range(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Analyse une expression de plage de dates.
        
        Args:
            text: Expression de plage (par exemple, "du 15 au 22 janvier")
            
        Returns:
            Tuple de (date_début, date_fin) au format ISO
        """
        # Plage française : du JJ au JJ mois
        range_match = re.search(
            r'du\s+(\d{1,2})\s+au\s+(\d{1,2})\s+([a-zéûô]+)(?:\s+(\d{4}))?',
            text.lower()
        )
        
        if range_match:
            start_day = int(range_match.group(1))
            end_day = int(range_match.group(2))
            month_name = range_match.group(3)
            year = int(range_match.group(4)) if range_match.group(4) else self.reference.year
            
            month = self.MONTHS_FR.get(month_name)
            if month:
                try:
                    start = datetime(year, month, start_day)
                    end = datetime(year, month, end_day)
                    logger.debug(f"Found French date range '{text}', returning ({start.strftime('%Y-%m-%d')}, {end.strftime('%Y-%m-%d')})")
                    return (start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
                except ValueError:
                    logger.debug(f"Invalid French date range: {range_match.groups()}")
                    pass
        
        logger.debug(f"No date range pattern matched for '{text}'")
        return (None, None)


if __name__ == "__main__":
    # Tester l'analyseur
    parser = DateParser()
    
    test_dates = [
        "demain",
        "le 15 janvier",
        "25 mars 2025",
        "dans 2 semaines",
        "lundi prochain",
        "15/03/2025",
        "2025-06-20",
        "tomorrow",
        "in 3 days",
    ]
    
    print("Test de l'analyseur de dates")
    print("=" * 50)
    print(f"Date de référence : {parser.reference.strftime('%Y-%m-%d')}")
    print()
    
    for expr in test_dates:
        result = parser.parse(expr)
        print(f"  '{expr}' -> {result}")
