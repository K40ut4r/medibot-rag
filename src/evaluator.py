from datasets import load_dataset
from src.rag_chain import build_rag_chain
from typing import Dict, List

def evaluate_on_pubmedqa(n_samples: int = 50) -> Dict:
    """
    Évalue le RAG sur un échantillon PubMedQA.
    Retourne des métriques simples : taux de réponse, citation des sources.
    """
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    samples = ds.select(range(min(n_samples, len(ds))))
    
    chain = build_rag_chain()
    results = []
    
    for item in samples:
        question = item["question"]
        # Contexte PubMed = la réponse attendue
        try:
            result = chain({"query": question})
            has_source = len(result.get("source_documents", [])) > 0
            results.append({
                "question": question,
                "has_answer": "Je ne dispose pas" not in result["result"],
                "has_source": has_source
            })
        except Exception as e:
            results.append({"question": question, "error": str(e)})
    
    # Métriques
    total = len(results)
    with_answer = sum(1 for r in results if r.get("has_answer", False))
    with_source = sum(1 for r in results if r.get("has_source", False))
    
    return {
        "total_evaluated": total,
        "answer_rate": with_answer / total,
        "source_citation_rate": with_source / total,
        "details": results
    }

if __name__ == "__main__":
    metrics = evaluate_on_pubmedqa(20)
    print(f"Taux de réponse : {metrics['answer_rate']:.2%}")
    print(f"Taux de citation : {metrics['source_citation_rate']:.2%}")