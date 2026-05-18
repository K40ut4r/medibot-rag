import re

URGENCY_PATTERNS = [
    r"crise cardiaque", r"arrêt cardiaque", r"ne respire plus",
    r"saignement", r"perte de connaissance", r"étouffement",
    r"empoisonnement", r"convulsions", r"brûlure grave", r"suicide"
]

def detect_intent(query: str):
    """L'agent comprend l'intention de l'utilisateur"""
    query_lower = query.lower()
    
    # 1. Détecter urgence
    for pattern in URGENCY_PATTERNS:
        if re.search(pattern, query_lower):
            return {
                "intent": "EMERGENCY",
                "message": "🔴 **URGENCE VITALE DÉTECTÉE :** S'il s'agit d'une urgence grave (douleur thoracique, étouffement, saignement), **APPELEZ IMMÉDIATEMENT LE 15 (SAMU) ou le 112.**",
                "action": "block_and_alert"
            }
    
    # 2. Détecter demande de RDV
    if any(word in query_lower for word in ["rdv", "rendez-vous", "prendre rdv", "voir un médecin"]):
        return {
            "intent": "APPOINTMENT",
            "message": None,
            "action": "suggest_appointment"
        }
    
    # 3. Détecter question médicale
    return {
        "intent": "MEDICAL_QUESTION",
        "message": None,
        "action": "rag_query"
    }