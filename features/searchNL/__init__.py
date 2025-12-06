# features/searchNL - Natural Language to XFT Query Transformation
"""
Transform natural language flight search queries into XFT XML format.
Uses qwen-14b via Ollama for entity extraction.
"""

from .nl_parser import NLParser
from .xft_builder import XFTBuilder
from .airport_db import AirportDatabase
from .date_parser import DateParser

__all__ = ['NLParser', 'XFTBuilder', 'AirportDatabase', 'DateParser']
__version__ = '1.0.0'
