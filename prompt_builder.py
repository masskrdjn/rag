# prompt_builder.py
from typing import List, Dict
from datetime import datetime

class PromptBuilder:
    """
    Construire des prompts structurés pour le LLM avec contexte enrichi.
    """
    
    SYSTEM_PROMPT_TEMPLATE = """Tu es un assistant expert en politique RH et systèmes de réservation (Amadeus/GDS).

INSTRUCTIONS CRITIQUES:
1. 🔒 Utilise UNIQUEMENT les informations des documents fournis ci-dessous
2. ❌ Si la réponse n'est pas dans les documents, dis-le explicitement et propose de consulter un expert
3. 📎 Cite TOUJOURS le document source et le numéro de section
4. 📋 Pour les procédures: numérote clairement chaque étape
5. 🔀 Si plusieurs documents traitent le sujet: synthétise les informations complémentaires
6. ⚠️ Confiance basse (<50%)? Inclus: "[⚠️ VÉRIFIER - Information peu sûre]"
7. 🚫 JAMAIS d'hallucinations ou de fabrication

FORMAT DE RÉPONSE ATTENDU:
- Réponse directe (1-2 paragraphes)
- Étapes numérotées si applicable
- Références avec [Ref #N] renvoyant à la section Documents
- Mise en garde si doute

CONTEXTE ACTUALISÉ: {timestamp}
"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def build_document_section(self, documents: List[Dict]) -> str:
        """Construire la section documents du contexte"""
        
        if not documents:
            return "## 📄 DOCUMENTS\n\n[Aucun document trouvé - Consulter un expert]"
        
        section = "## 📄 DOCUMENTS DE RÉFÉRENCE\n\n"
        
        for i, doc in enumerate(documents, 1):
            confidence = doc.get('rerank_score', 0.8)
            filename = doc.get('filename', 'N/A')
            section_title = doc.get('section_title', 'Section principale')
            doc_type = doc.get('doc_type', 'document').upper()
            content = doc.get('content', '')
            if not content and 'page_content' in doc:
                 content = doc['page_content']
            content = content[:400]
            
            # Indicateur visuel de confiance
            if confidence > 0.8:
                confidence_badge = "✅ HAUTE"
            elif confidence > 0.6:
                confidence_badge = "⚠️ MOYENNE"
            else:
                confidence_badge = "❌ BASSE"
            
            section += f"""### [REF #{i}] {filename}
**Type:** {doc_type} | **Confiance:** {confidence_badge} ({confidence:.0%})
**Section:** {section_title}

{content}...

---

"""
        
        return section
    
    def build_system_prompt(self, documents: List[Dict]) -> str:
        """Construire le prompt système complet"""
        
        doc_section = self.build_document_section(documents)
        
        return self.SYSTEM_PROMPT_TEMPLATE.format(
            timestamp=self.timestamp
        ) + "\n" + doc_section
    
    def build_user_prompt(self, question: str) -> str:
        """Construire le prompt utilisateur"""
        
        return f"""QUESTION: {question}

RÉPONSE (basée sur les références ci-dessus):"""
    
    def build_full_prompt(self, question: str, 
                         documents: List[Dict]) -> Dict:
        """Construire le prompt complet (système + utilisateur)"""
        
        return {
            'system': self.build_system_prompt(documents),
            'user': self.build_user_prompt(question),
            'full': (
                self.build_system_prompt(documents) + "\n" + 
                self.build_user_prompt(question)
            )
        }
