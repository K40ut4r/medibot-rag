from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from src.vector_store import load_vector_store
import yaml

MEDICAL_PROMPT = """<|system|>
Tu es un assistant médical. Réponds UNIQUEMENT à la question posée en te basant sur le contexte fourni. 
Ne répète JAMAIS le contexte dans ta réponse. Synthétise les informations en une réponse claire.
</|system|>

<|context|>
{context}
</|context|>

<|user|>
{question}
</|user|>

<|assistant|>
"""


def format_docs(docs):
    if not docs:
        return "Aucun document pertinent trouvé."
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Inconnu")
        formatted.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
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
        base_url="http://localhost:11434"
    )

    prompt = PromptTemplate(
        template=MEDICAL_PROMPT,
        input_variables=["context", "question"]
    )

    # ✅ Retriever simple et stable (sans score_threshold capricieux)
    retriever = vectordb.as_retriever(
        search_kwargs={"k": config['rag']['top_k']}
    )

    def retrieve_and_run(question: str) -> dict:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        return {
            "answer": answer,
            "source_documents": docs
        }

    return RunnableLambda(retrieve_and_run)