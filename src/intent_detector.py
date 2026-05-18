import re

URGENCY_PATTERNS = {
    "fr": [
        r"crise cardiaque", r"arrêt cardiaque", r"ne respire plus",
        r"saignement", r"perte de connaissance", r"étouffement",
        r"empoisonnement", r"convulsions", r"brûlure grave"
    ],
    "en": [
        r"heart attack", r"not breathing", r"unconscious",
        r"severe bleeding", r"choking", r"poisoning"
    ]
}

def detect_intent(query: str, lang: str = "fr"):
    """L'agent comprend l'intention de l'utilisateur"""
    query_lower = query.lower()
    
    # 1. Détecter urgence
    patterns = URGENCY_PATTERNS.get(lang, URGENCY_PATTERNS["fr"])
    for pattern in patterns:
        if re.search(pattern, query_lower):
            return {
                "intent": "EMERGENCY",
                "message": "🚨 APPELEZ IMMÉDIATEMENT LE 15 (SAMU) OU LE 112.",
                "action": "block_and_alert"
            }
    
    # 2. Détecter demande de RDV
    if any(word in query_lower for word in ["rdv", "rendez-vous", "appointment", "prendre rdv"]):
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