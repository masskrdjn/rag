#!/usr/bin/env python3
"""
Script de test comparatif pour optimiser la configuration du RAG.
Compare différentes stratégies de retrieval : k variable, score_threshold, et recherche hybride.
"""

from rag_pipeline import SimpleRAG
import time
import json
from datetime import datetime

# Questions de test réalistes (à adapter avec tes vraies questions métier)
TEST_QUESTIONS = [
    "Comment faire une demande de congés payés ?",
    "Quelle est la procédure pour les repas spécifiques ?",
    "Comment annuler un dossier GDS ?",
    "Que faire en cas de no-show d'un client ?",
    "Comment gérer une grève de la compagnie aérienne ?",
    "Quelle est la procédure pour les modifications de vol ?",
    "Comment traiter les demandes médicales ?",
    "Que faire pour les documents de voyage manquants ?",
    "Comment émettre une réservation IATA ?",
    "Quelle est la procédure de remboursement client ?",
]

def test_configuration(config_name, rag_instance, questions):
    """
    Teste une configuration et retourne les résultats.
    """
    print(f"\n{'='*80}")
    print(f"Test de la configuration: {config_name}")
    print(f"{'='*80}")
    
    results = {
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "retrieval_mode": rag_instance.retrieval_mode,
            "top_k": rag_instance.top_k,
            "score_threshold": rag_instance.score_threshold if hasattr(rag_instance, 'score_threshold') else None,
            "use_hybrid": rag_instance.use_hybrid,
            "hybrid_weights": rag_instance.hybrid_weights if rag_instance.use_hybrid else None,
        },
        "questions": []
    }
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: {question}")
        
        start_time = time.time()
        try:
            response = rag_instance.ask(question)
            elapsed_time = time.time() - start_time
            
            print(f"Réponse ({elapsed_time:.2f}s):")
            print(response[:300] + "..." if len(response) > 300 else response)
            
            results["questions"].append({
                "question": question,
                "response": response,
                "time_seconds": round(elapsed_time, 2),
                "response_length": len(response),
                "success": True
            })
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"ERREUR: {str(e)}")
            
            results["questions"].append({
                "question": question,
                "response": None,
                "error": str(e),
                "time_seconds": round(elapsed_time, 2),
                "success": False
            })
    
    # Calcul des statistiques
    successful_queries = [q for q in results["questions"] if q["success"]]
    if successful_queries:
        avg_time = sum(q["time_seconds"] for q in successful_queries) / len(successful_queries)
        avg_length = sum(q["response_length"] for q in successful_queries) / len(successful_queries)
        
        results["statistics"] = {
            "total_questions": len(questions),
            "successful_queries": len(successful_queries),
            "failed_queries": len(questions) - len(successful_queries),
            "avg_response_time": round(avg_time, 2),
            "avg_response_length": round(avg_length, 0)
        }
        
        print(f"\n{'='*40}")
        print(f"Statistiques:")
        print(f"  - Questions réussies: {len(successful_queries)}/{len(questions)}")
        print(f"  - Temps moyen: {avg_time:.2f}s")
        print(f"  - Longueur moyenne: {avg_length:.0f} caractères")
    
    return results

def main():
    """
    Lance les tests sur différentes configurations.
    """
    print("="*80)
    print("BENCHMARK DES CONFIGURATIONS DE RETRIEVAL")
    print("="*80)
    
    all_results = []
    
    # Configuration 1: Baseline (similarity avec k=5)
    print("\n📊 Configuration 1: Baseline (similarity, k=5)")
    rag1 = SimpleRAG(retrieval_mode="similarity", top_k=5)
    results1 = test_configuration("baseline_k5", rag1, TEST_QUESTIONS)
    all_results.append(results1)
    
    # Configuration 2: k=3 (moins de contexte)
    print("\n📊 Configuration 2: Similarity avec k=3")
    rag2 = SimpleRAG(retrieval_mode="similarity", top_k=3)
    results2 = test_configuration("similarity_k3", rag2, TEST_QUESTIONS)
    all_results.append(results2)
    
    # Configuration 3: k=8 (plus de contexte)
    print("\n📊 Configuration 3: Similarity avec k=8")
    rag3 = SimpleRAG(retrieval_mode="similarity", top_k=8)
    results3 = test_configuration("similarity_k8", rag3, TEST_QUESTIONS)
    all_results.append(results3)
    
    # Configuration 4: Score threshold 0.7
    print("\n📊 Configuration 4: Score threshold 0.7 (k=8)")
    rag4 = SimpleRAG(retrieval_mode="similarity_score_threshold", top_k=8, score_threshold=0.7)
    results4 = test_configuration("threshold_0.7", rag4, TEST_QUESTIONS)
    all_results.append(results4)
    
    # Configuration 5: Score threshold 0.5 (plus permissif)
    print("\n📊 Configuration 5: Score threshold 0.5 (k=8)")
    rag5 = SimpleRAG(retrieval_mode="similarity_score_threshold", top_k=8, score_threshold=0.5)
    results5 = test_configuration("threshold_0.5", rag5, TEST_QUESTIONS)
    all_results.append(results5)
    
    # Configuration 6: Recherche hybride (50/50)
    print("\n📊 Configuration 6: Hybride BM25+Vector (50/50, k=5)")
    rag6 = SimpleRAG(retrieval_mode="similarity", top_k=5, use_hybrid=True, hybrid_weights=[0.5, 0.5])
    results6 = test_configuration("hybrid_50_50", rag6, TEST_QUESTIONS)
    all_results.append(results6)
    
    # Configuration 7: Recherche hybride (70% vector, 30% BM25)
    print("\n📊 Configuration 7: Hybride BM25+Vector (70/30, k=5)")
    rag7 = SimpleRAG(retrieval_mode="similarity", top_k=5, use_hybrid=True, hybrid_weights=[0.7, 0.3])
    results7 = test_configuration("hybrid_70_30", rag7, TEST_QUESTIONS)
    all_results.append(results7)
    
    # Sauvegarder tous les résultats
    output_file = f"retrieval_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Benchmark terminé ! Résultats sauvegardés dans: {output_file}")
    print(f"{'='*80}")
    
    # Afficher un résumé comparatif
    print("\n📈 RÉSUMÉ COMPARATIF:")
    print(f"{'='*80}")
    print(f"{'Configuration':<25} | {'Réussis':>8} | {'Temps':>8} | {'Longueur':>10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        if "statistics" in result:
            stats = result["statistics"]
            print(f"{result['config_name']:<25} | {stats['successful_queries']:>3}/{stats['total_questions']:>3} | "
                  f"{stats['avg_response_time']:>6.2f}s | {stats['avg_response_length']:>8.0f} car")
    
    print(f"{'='*80}")
    print("\n💡 RECOMMANDATIONS:")
    print("1. Compare les réponses manuellement pour évaluer la qualité")
    print("2. Vérifie si certaines configs produisent trop de bruit ou pas assez de contexte")
    print("3. La meilleure config = meilleur équilibre précision/pertinence")
    print("4. Si threshold rejette trop de questions, diminue le seuil")
    print("5. Si hybrid améliore les résultats, ajuste les weights")

if __name__ == "__main__":
    main()
