from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from src.vector_store import load_vector_store
import yaml
from langdetect import detect

MULTILINGUAL_PROMPT = """<<||<|<|system|>
Tu es un assistant médical intelligent. Réponds UNIQUEMENT à la question posée ci-dessous.
Si le contexte ne permet pas de répondre précisément, dis-le clairement.
Ne fais PAS de suppositions sur le profil patient (diabétique ou non) sauf si explicitement mentionné.
Réponds dans la langue de la question ({langue}).
Sois concis : 3 à 5 phrases maximum.

<<||<|<|context|>
{context}

<<||<|<|user|>
{question}

<<||<|<|assistant|>
"""

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "ar": 
            return "arabe"
        if lang == "en": 
            return "anglais"
        return "français"
    except Exception:
        return "français"

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

    def retrieve_and_run(question: str) -> dict:
        langue = detect_language(question)
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        prompt = PromptTemplate(
            template=MULTILINGUAL_PROMPT,
            input_variables=["context", "question", "langue"]
        )
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context, 
            "question": question,
            "langue": langue
        })
        
        return {
            "answer": answer,
            "source_documents": docs,
            "detected_language": langue
        }

    return RunnableLambda(retrieve_and_run)