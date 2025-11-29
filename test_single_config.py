#!/usr/bin/env python3
"""
Script simple pour tester rapidement une configuration spécifique du RAG.
Usage: python test_single_config.py
"""

from rag_pipeline import SimpleRAG

def main():
    """
    Teste une configuration spécifique.
    Modifie les paramètres ci-dessous pour tester différentes configs.
    """
    
    # ========== CONFIGURATION À TESTER ==========
    # Décommente la configuration que tu veux tester:
    
    # Option 1: Similarity avec k variable
    rag = SimpleRAG(retrieval_mode="similarity", top_k=5)
    
    # Option 2: Similarity avec score threshold
    # rag = SimpleRAG(retrieval_mode="similarity_score_threshold", top_k=8, score_threshold=0.7)
    
    # Option 3: Recherche hybride (BM25 + vector)
    # rag = SimpleRAG(use_hybrid=True, top_k=5, hybrid_weights=[0.5, 0.5])
    
    # Option 4: Hybride avec plus de poids sur les vecteurs
    # rag = SimpleRAG(use_hybrid=True, top_k=5, hybrid_weights=[0.7, 0.3])
    
    # ============================================
    
    print("="*80)
    print("TEST DE CONFIGURATION RAG")
    print("="*80)
    print(f"Mode de retrieval: {rag.retrieval_mode}")
    print(f"Top-k: {rag.top_k}")
    if rag.retrieval_mode == "similarity_score_threshold":
        print(f"Score threshold: {rag.score_threshold}")
    if rag.use_hybrid:
        print(f"Recherche hybride activée")
        print(f"Weights (vector, BM25): {rag.hybrid_weights}")
    print("="*80)
    
    # Questions de test
    questions = [
        "Comment faire une demande de congés payés ?",
        "Quelle est la procédure pour les repas spécifiques ?",
        "Comment annuler un dossier GDS ?",
        "Que faire en cas de no-show ?",
        "Comment gérer une grève de la compagnie ?",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"{'='*80}")
        
        try:
            response = rag.ask(question)
            print(response)
        except Exception as e:
            print(f"ERREUR: {str(e)}")
    
    print(f"\n{'='*80}")
    print("✅ Test terminé")
    print("="*80)

if __name__ == "__main__":
    main()
