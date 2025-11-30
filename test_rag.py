#!/usr/bin/env python3
"""
Script de test complet pour le système RAG.
Interroge l'API question par question et log les résultats dans un fichier texte.
"""

import json
import requests
import time
from datetime import datetime
from pathlib import Path
import sys

# Configuration
API_URL = "http://localhost:8000/ask"
QUESTIONS_FILE = "test_questions.json"
RESULTS_FILE = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
DELAY_BETWEEN_REQUESTS = 1  # secondes entre chaque requête

def load_questions(filepath: str) -> list:
    """Charge les questions depuis le fichier JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier {filepath} introuvable")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Erreur: Fichier JSON invalide - {e}")
        sys.exit(1)

def query_rag(question: str) -> dict:
    """Interroge l'API RAG avec une question."""
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=120
        )
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
            "status_code": response.status_code
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Impossible de se connecter à l'API (connexion refusée)",
            "status_code": 0
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Timeout - l'API a mis trop de temps à répondre",
            "status_code": 0
        }
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "error": f"Erreur HTTP: {e}",
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Erreur inattendue: {str(e)}",
            "status_code": 0
        }

def format_sources(sources: list) -> str:
    """Formate les sources pour l'affichage."""
    if not sources:
        return "Aucune source"
    
    formatted = []
    for i, source in enumerate(sources, 1):
        # Utiliser les bons champs retournés par l'API
        title = source.get('title', 'N/A')
        source_url = source.get('source_url', '')
        category = source.get('category', 'Unknown')
        post_id = source.get('post_id', '')
        
        source_info = f"  [{i}] {title}"
        if source_url:
            source_info += f"\n      URL: {source_url}"
        if category and category != 'Unknown':
            source_info += f"\n      Catégorie: {category}"
        if post_id:
            source_info += f" (ID: {post_id})"
        formatted.append(source_info)
    
    return "\n".join(formatted)

def write_result(file, category: str, question: str, result: dict, question_num: int, total: int):
    """Écrit un résultat dans le fichier de log."""
    separator = "=" * 80
    
    file.write(f"\n{separator}\n")
    file.write(f"Question {question_num}/{total}\n")
    file.write(f"Catégorie: {category}\n")
    file.write(f"{separator}\n\n")
    file.write(f"QUESTION:\n{question}\n\n")
    
    if result["success"]:
        data = result["data"]
        file.write(f"RÉPONSE:\n{data.get('answer', 'N/A')}\n\n")
        
        if 'sources' in data and data['sources']:
            file.write(f"SOURCES ({len(data['sources'])}):\n")
            file.write(format_sources(data['sources']))
            file.write("\n\n")
        else:
            file.write("SOURCES:\nAucune source trouvée\n\n")
        
        if 'metadata' in data:
            file.write(f"MÉTADONNÉES:\n{json.dumps(data['metadata'], indent=2, ensure_ascii=False)}\n\n")
    else:
        file.write(f"❌ ERREUR:\n{result['error']}\n")
        file.write(f"Status Code: {result['status_code']}\n\n")
    
    file.flush()  # Force l'écriture immédiate

def print_progress(question_num: int, total: int, category: str, success: bool):
    """Affiche la progression dans le terminal."""
    status = "✓" if success else "✗"
    percentage = (question_num / total) * 100
    print(f"[{percentage:5.1f}%] {status} Question {question_num}/{total} - {category}")

def run_tests():
    """Exécute tous les tests."""
    print("🚀 Démarrage des tests du RAG")
    print(f"📁 Chargement des questions depuis: {QUESTIONS_FILE}")
    
    categories = load_questions(QUESTIONS_FILE)
    
    # Compte total des questions
    total_questions = sum(len(cat['questions']) for cat in categories)
    print(f"📊 {total_questions} questions trouvées dans {len(categories)} catégories")
    print(f"📝 Les résultats seront sauvegardés dans: {RESULTS_FILE}\n")
    
    # Vérifie la connexion à l'API
    print("🔌 Vérification de la connexion à l'API...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✓ API accessible\n")
        else:
            print(f"⚠ API répond avec le code: {response.status_code}\n")
    except Exception as e:
        print(f"⚠ Impossible de vérifier l'API: {e}")
        print("⚠ Les tests vont continuer mais pourraient échouer\n")
    
    # Initialisation des compteurs
    current_question = 0
    success_count = 0
    error_count = 0
    start_time = time.time()
    
    # Création du fichier de résultats
    with open(RESULTS_FILE, 'w', encoding='utf-8') as log_file:
        # En-tête du fichier
        log_file.write("=" * 80 + "\n")
        log_file.write("RÉSULTATS DES TESTS RAG\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total de questions: {total_questions}\n")
        log_file.write(f"API URL: {API_URL}\n")
        log_file.write("=" * 80 + "\n")
        
        # Traitement des questions
        for category_data in categories:
            category = category_data['category']
            questions = category_data['questions']
            
            for question in questions:
                current_question += 1
                
                # Interroge l'API
                result = query_rag(question)
                
                # Mise à jour des compteurs
                if result["success"]:
                    success_count += 1
                else:
                    error_count += 1
                
                # Écrit dans le fichier
                write_result(log_file, category, question, result, current_question, total_questions)
                
                # Affiche la progression
                print_progress(current_question, total_questions, category, result["success"])
                
                # Délai entre les requêtes (sauf pour la dernière)
                if current_question < total_questions:
                    time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Résumé final
        elapsed_time = time.time() - start_time
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("RÉSUMÉ FINAL\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Total questions: {total_questions}\n")
        log_file.write(f"Succès: {success_count} ({success_count/total_questions*100:.1f}%)\n")
        log_file.write(f"Erreurs: {error_count} ({error_count/total_questions*100:.1f}%)\n")
        log_file.write(f"Temps total: {elapsed_time:.2f} secondes\n")
        log_file.write(f"Temps moyen par question: {elapsed_time/total_questions:.2f} secondes\n")
        log_file.write("=" * 80 + "\n")
    
    # Affichage du résumé
    print("\n" + "=" * 80)
    print("✅ TESTS TERMINÉS")
    print("=" * 80)
    print(f"📊 Résultats:")
    print(f"   Total: {total_questions} questions")
    print(f"   ✓ Succès: {success_count} ({success_count/total_questions*100:.1f}%)")
    print(f"   ✗ Erreurs: {error_count} ({error_count/total_questions*100:.1f}%)")
    print(f"⏱  Temps total: {elapsed_time:.2f} secondes")
    print(f"⏱  Temps moyen: {elapsed_time/total_questions:.2f} secondes/question")
    print(f"📝 Résultats détaillés: {RESULTS_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
