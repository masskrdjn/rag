# api_handler.py
"""
Point de terminaison FastAPI pour SearchNL - Transformation du langage naturel en XFT.
Fournit une API REST pour l'intégration du backend PHP.
"""
import os
import sys
import logging

# Forcer l'utilisation du CPU pour la cohérence
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "api_handler.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn

from .config_searchnl import API_CONFIG, MODEL_NAME
from .nl_parser import NLParser, FlightQuery
from .xft_builder import XFTBuilder


# =============================================================================
# MODÈLES D'API
# =============================================================================

class NLRequest(BaseModel):
    """Modèle de requête pour l'analyse du langage naturel."""
    query: str = Field(..., description="Requête de recherche de vol en langage naturel")
    use_llm: bool = Field(True, description="Utiliser le LLM pour l'analyse (retour au regex si Faux)")
    prefer_metropolitan: bool = Field(True, description="Préférer les codes de ville aux codes d'aéroport")


class ParsedEntities(BaseModel):
    """Entités de vol analysées."""
    origin: str
    origin_code: str
    destination: str
    destination_code: str
    departure_date: str
    return_date: Optional[str] = None
    adults: int = 1
    children: int = 0
    infants: int = 0
    trip_type: str = "OW"
    cabin_class: str = "economy"
    flexible_dates: bool = False
    direct_only: bool = False


class XFTResponse(BaseModel):
    """Modèle de réponse avec XML XFT et entités analysées."""
    success: bool
    xft_xml: Optional[str] = None
    parsed_entities: Optional[ParsedEntities] = None
    error: Optional[str] = None
    raw_query: str = ""


class HealthResponse(BaseModel):
    """Réponse de vérification de l'état de santé."""
    status: str
    model: str
    version: str


# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="API SearchNL",
    description="Transformation de la requête en langage naturel vers XFT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware CORS pour l'intégration du backend PHP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configurer de manière appropriée pour la production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instances globales
nl_parser: Optional[NLParser] = None
xft_builder: Optional[XFTBuilder] = None


@app.on_event("startup")
async def startup_event():
    """Initialiser l'analyseur et le constructeur au démarrage."""
    global nl_parser, xft_builder
    
    logger.info("🚀 Démarrage de l'API SearchNL...")
    logger.info(f"Modèle configuré: {MODEL_NAME}")
    logger.info(f"Port configuré: {API_CONFIG['port']}")
    
    print("🚀 Démarrage de l'API SearchNL...")
    print(f"   Modèle: {MODEL_NAME}")
    print(f"   Port: {API_CONFIG['port']}")
    
    try:
        logger.debug("Initialisation de NLParser...")
        nl_parser = NLParser(use_llm=True)
        logger.debug("NLParser initialisé")
        
        logger.debug("Initialisation de XFTBuilder...")
        xft_builder = XFTBuilder()
        logger.debug("XFTBuilder initialisé")
        
        logger.info("✓ API SearchNL prête")
        print("✓ API SearchNL prête")
    except Exception as e:
        logger.error(f"✗ Échec de l'initialisation: {e}")
        print(f"✗ Échec de l'initialisation: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Point de terminaison de vérification de l'état de santé."""
    status = "sain" if nl_parser else "initialisation"
    logger.debug(f"Health check: status={status}")
    return HealthResponse(
        status=status,
        model=MODEL_NAME,
        version="1.0.0"
    )


@app.post("/parse", response_model=XFTResponse)
async def parse_query(request: NLRequest):
    """
    Analyse une requête en langage naturel et génère du XML XFT.
    
    Exemples de requêtes:
    - "vol Paris Londres demain 2 adultes"
    - "Je cherche un aller-retour Nice Rome du 15 au 22 mars"
    - "billet New York 25/12/2025 1 adulte 1 enfant business"
    """
    logger.info(f"Requête reçue: '{request.query}'")
    logger.debug(f"Options: use_llm={request.use_llm}, prefer_metropolitan={request.prefer_metropolitan}")
    
    if not nl_parser or not xft_builder:
        logger.error("Service non prêt - nl_parser ou xft_builder non initialisé")
        raise HTTPException(status_code=503, detail="Service non prêt")
    
    if not request.query.strip():
        logger.warning("Requête vide reçue")
        return XFTResponse(
            success=False,
            error="La requête ne peut pas être vide",
            raw_query=request.query
        )
    
    try:
        # Analyser le langage naturel
        logger.debug("Début de l'analyse du langage naturel")
        flight_query = nl_parser.parse(request.query)
        
        if not flight_query:
            logger.warning(f"Analyse échouée pour: '{request.query}'")
            return XFTResponse(
                success=False,
                error="Impossible d'analyser la requête - origine ou destination manquante",
                raw_query=request.query
            )
        
        logger.info(f"Analyse réussie: {flight_query.origin_code} -> {flight_query.destination_code}")
        
        # Construire le XML XFT
        logger.debug("Construction du XML XFT")
        xft_xml = xft_builder.build_from_query(flight_query)
        
        # Valider le XML
        logger.debug("Validation du XML généré")
        if not xft_builder.validate(xft_xml):
            logger.error("Le XML généré a échoué à la validation")
            return XFTResponse(
                success=False,
                error="Le XML généré a échoué à la validation",
                raw_query=request.query
            )
        
        logger.info(f"XFT généré avec succès pour: '{request.query}'")
        return XFTResponse(
            success=True,
            xft_xml=xft_xml,
            parsed_entities=ParsedEntities(
                origin=flight_query.origin,
                origin_code=flight_query.origin_code,
                destination=flight_query.destination,
                destination_code=flight_query.destination_code,
                departure_date=flight_query.departure_date,
                return_date=flight_query.return_date,
                adults=flight_query.adults,
                children=flight_query.children,
                infants=flight_query.infants,
                trip_type=flight_query.trip_type,
                cabin_class=flight_query.cabin_class,
                flexible_dates=flight_query.flexible_dates,
                direct_only=flight_query.direct_only,
            ),
            raw_query=request.query
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {e}", exc_info=True)
        return XFTResponse(
            success=False,
            error=str(e),
            raw_query=request.query
        )


@app.post("/validate")
async def validate_xft(xft_xml: str):
    """Valider la structure XML XFT."""
    logger.debug("Requête de validation XFT reçue")
    if not xft_builder:
        logger.error("Service non prêt pour validation")
        raise HTTPException(status_code=503, detail="Service non prêt")
    
    is_valid = xft_builder.validate(xft_xml)
    logger.info(f"Résultat validation XFT: {'valide' if is_valid else 'invalide'}")
    return {"valid": is_valid}


@app.get("/airports/search")
async def search_airports(q: str, limit: int = 5):
    """Rechercher des aéroports par nom ou code."""
    logger.debug(f"Recherche aéroports: q='{q}', limit={limit}")
    if not nl_parser:
        logger.error("Service non prêt pour recherche aéroports")
        raise HTTPException(status_code=503, detail="Service non prêt")
    
    results = nl_parser.airport_db.search(q, limit=limit)
    logger.info(f"Recherche aéroports '{q}': {len(results)} résultat(s) trouvé(s)")
    
    return {
        "query": q,
        "results": [
            {
                "code": code,
                "name": info.name,
                "city": info.city,
                "country": info.country,
                "is_metropolitan": info.is_metropolitan,
                "score": score
            }
            for code, info, score in results
        ]
    }


# =============================================================================
# PRINCIPAL
# =============================================================================

def run_server():
    """Exécuter le serveur API."""
    logger.info(f"Lancement du serveur sur {API_CONFIG['host']}:{API_CONFIG['port']}")
    uvicorn.run(
        "features.searchNL.api_handler:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["debug"],
    )


if __name__ == "__main__":
    # Permettre l'exécution directe
    logger.info(f"Exécution directe du serveur sur {API_CONFIG['host']}:{API_CONFIG['port']}")
    uvicorn.run(
        app,
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
    )
