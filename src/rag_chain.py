from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from src.vector_store import load_vector_store
import yaml

BASIC_PROMPT_WITH_HISTORY = """<<||<|<|system|>
Tu es un assistant médical intelligent. Réponds UNIQUEMENT à la question posée ci-dessous.
Si le contexte ne permet pas de répondre précisément, dis-le clairement.
Tiens compte de l'historique de la conversation pour répondre.
Ne fais PAS de suppositions sur le profil patient sauf si explicitement mentionné.
Réponds en français.
Sois concis : 3 à 5 phrases maximum.

<<||<|<|historique|>
{history}

<<||<|<|context|>
{context}

<<||<|<|user|>
{question}

<<||<|<|assistant|>
"""

def format_docs(docs):
    if not docs:
        return "Aucun document pertinent trouvé."
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Inconnu")
        section = doc.metadata.get("section", "Non classé")
        # Tronquer pour accélérer le LLM (évite les réponses trop longues)
        content = doc.page_content[:600]
        formatted.append(f"[Source {i+1}: {source} | Section: {section}]\n{content}")
    return "\n\n---\n\n".join(formatted)

def build_rag_chain(config_path: str = "config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("🔍 Chargement de la base vectorielle...")
    vectordb = load_vector_store(
        config['paths']['faiss_index'],
        config['rag']['model_name']
    )

    print("🤖 Connexion à Ollama...")
    llm = Ollama(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        base_url="http://localhost:11434",
        keep_alive="30m",       # ⭐ Garde le modèle en RAM 30 min
        num_ctx=2048,           # ⭐ Limite le contexte
        num_predict=256,        # ⭐ Force des réponses courtes
    )

    # ⭐ Réduire top_k à 3 max pour aller plus vite
    retriever = vectordb.as_retriever(
        search_kwargs={"k": min(config['rag'].get('top_k', 3), 3)}
    )

    def retrieve_and_run(inputs: dict) -> dict:
        question = inputs.get("question", "")
        history_list = inputs.get("history", [])
        
        # 1. Préparer l'historique sous forme de texte
        history_str = "Aucun historique."
        search_query = question
        
        if history_list:
            history_str = ""
            for turn in history_list:
                history_str += f"Patient: {turn['question']}\nAssistant: {turn['answer']}\n\n"
            
            # 2. Améliorer la recherche : on ajoute la dernière question au search_query
            last_question = history_list[-1]["question"]
            search_query = f"{last_question} {question}"
            
        # Retrieval avec la requête contextualisée
        docs = retriever.invoke(search_query)
        context = format_docs(docs)
        
        prompt = PromptTemplate(
            template=BASIC_PROMPT_WITH_HISTORY,
            input_variables=["history", "context", "question"]
        )
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "history": history_str,
            "context": context, 
            "question": question
        })
        
        return {
            "answer": answer,
            "source_documents": docs
        }

    return RunnableLambda(retrieve_and_run)