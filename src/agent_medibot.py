from src.rag_chain import build_rag_chain
from src.intent_detector import detect_intent

class MediBotAgent:
    def __init__(self):
        self.rag_chain = build_rag_chain()
        self.conversation_history = []
    
    def run(self, query: str):
        """L'agent raisonne et choisit l'action"""
        
        # ÉTAPE 1 : Détecter l'intention (l'agent "réfléchit")
        intent = detect_intent(query)
        
        if intent["intent"] == "EMERGENCY":
            return {
                "answer": intent["message"],
                "sources": [],
                "is_emergency": True,
                "suggested_action": "call_samu"
            }
        
        if intent["intent"] == "APPOINTMENT":
            return {
                "answer": "Je peux vous aider à prendre rendez-vous. Voici le lien vers l'application : [Prendre un rendez-vous](https://github.com/OssamaAgourari/HealthManagment)",
                "sources": [],
                "is_emergency": False,
                "suggested_action": "open_app"
            }
        
        # ÉTAPE 2 : Exécuter RAG (l'agent "cherche" en passant l'historique)
        result = self.rag_chain.invoke({
            "question": query,
            "history": self.conversation_history[-3:] # On garde les 3 derniers échanges max
        })
        
        # ÉTAPE 3 : Formater pour un patient unique
        answer = result["answer"] + "\n\n⚕️ N'oubliez pas : cette information ne remplace pas un avis médical."
        
        # ÉTAPE 4 : Sauvegarder mémoire (l'agent "apprend")
        self.conversation_history.append({"question": query, "answer": result["answer"]})
        
        return {
            "answer": answer,
            "sources": result.get("source_documents", []),
            "is_emergency": False
        }