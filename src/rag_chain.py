from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.vector_store import load_vector_store
import yaml

MEDICAL_PROMPT = """Tu es un assistant médical intelligent. Tu réponds UNIQUEMENT à partir du contexte médical fourni.
Si tu ne trouves pas la réponse dans le contexte, dis "Je ne dispose pas d'informations suffisantes dans ma base de connaissances pour répondre à cette question."

Contexte médical :
{context}

Question du patient : {question}

Instructions :
1. Réponds de manière claire et structurée
2. Cite les sources entre crochets [source: nom_du_document]
3. Ajoute TOUJOURS un avertissement médical à la fin
4. Si c'est une urgence potentielle, conseille de consulter immédiatement

Réponse :"""

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
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": config['rag']['top_k']}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain