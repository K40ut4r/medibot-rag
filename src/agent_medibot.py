from src.rag_chain import build_rag_chain
from src.intent_detector import detect_intent

class MediBotAgent:
    def __init__(self):
        self.rag_chain = build_rag_chain()
        self.conversation_history = []
    
    def run(self, query: str, mode: str = "patient", lang: str = "fr"):
        """L'agent raisonne et choisit l'action"""
        
        # ÉTAPE 1 : Détecter l'intention (l'agent "réfléchit")
        intent = detect_intent(query, lang)
        
        if intent["intent"] == "EMERGENCY":
            return {
                "answer": intent["message"],
                "sources": [],
                "is_emergency": True,
                "suggested_action": "call_samu"
            }
        
        if intent["intent"] == "APPOINTMENT":
            return {
                "answer": "Je peux vous aider à prendre rendez-vous. Voici le lien vers l'application :",
                "sources": [],
                "is_emergency": False,
                "suggested_action": "open_app",
                "app_link": "https://github.com/OssamaAgourari/HealthManagment"
            }
        
        # ÉTAPE 2 : Exécuter RAG (l'agent "cherche")
        result = self.rag_chain({
            "question": query,
            "chat_history": self.conversation_history
        })
        
        # ÉTAPE 3 : Formater selon le mode (l'agent "s'adapte")
        answer = self._format_by_mode(result["answer"], mode)
        
        # ÉTAPE 4 : Sauvegarder mémoire (l'agent "apprend")
        self.conversation_history.append((query, answer))
        
        return {
            "answer": answer,
            "sources": result.get("source_documents", []),
            "is_emergency": False,
            "mode": mode
        }
    
    def _format_by_mode(self, answer: str, mode: str):
        if mode == "patient":
            return answer + "\n\n⚕️ N'oubliez pas : cette information ne remplace pas un médecin."
        elif mode == "student":
            return answer + "\n\n📚 Référez-vous aux articles cités pour approfondir."
        elif mode == "doctor":
            return answer + "\n\n🔬 Voir les études complètes dans les sources ci-dessus."
        return answer