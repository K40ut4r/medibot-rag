import json
from src.rag_chain import build_rag_chain

TEST_SET = [
    {
        "question": "Quels sont les symptômes du diabète de type 2 ?",
        "expected_sources": ["diabete", "parcours"],
        "expected_keywords": ["soif", "polyurie", "fatigue", "vision"],
        "category": "Endocrinologie"
    },
    {
        "question": "What are the symptoms of type 2 diabetes?",
        "expected_sources": ["diabete", "parcours"],
        "expected_keywords": ["thirst", "urination", "fatigue"],
        "category": "Endocrinologie"
    },
    {
        "question": "ما هي أعراض السكري من النوع الثاني؟",
        "expected_sources": ["diabete", "parcours"],
        "expected_keywords": [],  # Arabe : on vérifie juste le retrieval
        "category": "Endocrinologie"
    },
    {
        "question": "Comment prévenir l'hypertension artérielle ?",
        "expected_sources": ["hypertension", "pression"],
        "expected_keywords": ["sport", "sel", "alimentation", "poids"],
        "category": "Cardiologie"
    },
    {
        "question": "Quels sont les effets secondaires du vaccin contre la grippe ?",
        "expected_sources": ["vaccin", "grippe", "influenza"],
        "expected_keywords": ["fièvre", "douleur", "injection", "bras"],
        "category": "Immunologie"
    },
    {
        "question": "Comment traiter un ulcère gastrique ?",
        "expected_sources": ["ulcere", "gastrique", "helicobacter"],
        "expected_keywords": ["antibiotique", "ppi", "oméprazole", "bismuth"],
        "category": "Gastro-entérologie"
    },
    {
        "question": "Quels sont les signes d'un asthme chez l'enfant ?",
        "expected_sources": ["asthme", "pediatrie", "enfant"],
        "expected_keywords": ["toux", "sifflement", "dyspnee", "crise"],
        "category": "Pédiatrie"
    },
    {
        "question": "What causes asthma in children?",
        "expected_sources": ["asthme", "pediatrie"],
        "expected_keywords": ["allergy", "trigger", "wheezing", "cough"],
        "category": "Pédiatrie"
    },
    {
        "question": "Comment réduire le cholestérol LDL ?",
        "expected_sources": ["cholesterol", "ldl", "statine"],
        "expected_keywords": ["statine", "régime", "exercice", "lipide"],
        "category": "Cardiologie"
    },
    {
        "question": "Quand consulter pour une douleur thoracique ?",
        "expected_sources": ["douleur", "thoracique", "cardiaque"],
        "expected_keywords": ["urgence", "infarctus", "angine", "consultation"],
        "category": "Cardiologie"
    }
]

def run_benchmark():
    chain = build_rag_chain()
    results = []
    
    for item in TEST_SET:
        result = chain.invoke(item["question"])
        answer = result["answer"].lower()
        sources = [d.metadata.get("source", "").lower() for d in result.get("source_documents", [])]
        
        # Vérifie si au moins une source attendue est présente
        retrieval_ok = any(
            any(exp in src for src in sources) 
            for exp in item["expected_sources"]
        ) if sources else False
        
        # Vérifie les mots-clés (sauf pour l'arabe où on skip)
        if item["expected_keywords"]:
            answer_ok = sum(1 for kw in item["expected_keywords"] if kw in answer) >= len(item["expected_keywords"])//2 + 1
        else:
            answer_ok = len(answer) > 50  # Vérifie juste qu'il y a une réponse
        
        results.append({
            "question": item["question"][:60] + "...",
            "category": item["category"],
            "langue": result.get("detected_language", "français"),
            "retrieval_ok": retrieval_ok,
            "answer_ok": answer_ok,
            "sources_found": [d.metadata.get("source", "Inconnu") for d in result.get("source_documents", [])]
        })
    
    # Sauvegarde JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

if __name__ == "__main__":
    run_benchmark()