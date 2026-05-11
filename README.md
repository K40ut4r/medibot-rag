# 🩺 MediBot RAG — Assistant Médical Intelligent

&gt; Projet de fin de module NLP & RAG — 4ème année Ingénierie AI/ML

MediBot est un agent conversationnel médical basé sur **RAG (Retrieval-Augmented Generation)** qui répond aux questions de santé en s'appuyant sur des sources scientifiques fiables (HAS, WHO, guides cliniques). Contrairement aux LLM classiques, MediBot **cite toujours ses sources** et inclut des avertissements médicaux.

![Interface MediBot](docs/screenshot.png)

---

## 🎯 Fonctionnalités

- ✅ **RAG médical** : Réponses basées sur des PDFs médicaux (guides HAS, OMS, etc.)
- ✅ **Citations de sources** : Chaque réponse indique le document source
- ✅ **Avertissement médical** automatique
- ✅ **Interface Streamlit** intuitive et responsive
- ✅ **LLM local** (Ollama) — 100% offline, pas de données envoyées dans le cloud
- ✅ **Embeddings sémantiques** (Sentence Transformers) pour la recherche vectorielle

---

## 🏗️ Architecture
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Streamlit │────→│  RAG Chain  │────→│   Ollama    │
│     UI      │     │ (LangChain) │     │   (LLM)     │
└─────────────┘     └──────┬──────┘     └─────────────┘
│
┌──────┴──────┐
│  FAISS DB   │
│  (Vectors)  │
└──────┬──────┘
│
┌──────┴──────┐
│  PDFs médicaux│
│  (Sources)    │
└─────────────┘


---

## 🚀 Installation

### Prérequis
- Python 3.11+
- [Ollama](https://ollama.com/) installé
- 4GB+ RAM disponible

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/medibot-rag.git
cd medibot-rag
2. Environnement virtuel

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

3. Installer les dépendances
pip install -r requirements.txt

4. Télécharger un modèle Ollama
ollama pull phi3:mini
# ou mistral si vous avez plus de RAM

5. Ajouter des documents médicaux
Placez vos PDFs (guides HAS, OMS, etc.) dans le dossier data/raw/

6. Ingestion (création de la base vectorielle)
python ingest.py

7. Lancer l'application
streamlit run app.py
L'application sera accessible sur http://localhost:8501

📁 Structure du projet

medibot-rag/
├── app.py                    # Interface Streamlit
├── ingest.py                 # Script d'ingestion des PDFs
├── config.yaml               # Configuration
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Extraction PDF
│   ├── chunker.py            # Découpage des textes
│   ├── vector_store.py       # Base FAISS + embeddings
│   └── rag_chain.py          # Chaîne RAG complète
└── data/
    ├── raw/                  # PDFs source (non versionnés)
    └── faiss_index/          # Base vectorielle générée

🧪 Exemple d'utilisation
Question : "Quels sont les symptômes du diabète ?"
Réponse :
Les symptômes du diabète incluent généralement des signes tels que la fatigue et l'augmentation de la soif. D'autres manifestations peuvent comprendre une augmentation de la fréquence urinaire...
📚 Source : HAS_Parcours_soins_diabete_type2.pdf

🛡️ Avertissement
Ce projet est à but éducatif uniquement. Les informations fournies ne remplacent en aucun cas un avis médical professionnel. En cas d'urgence, contactez le SAMU (15) ou les urgences.
👨‍💻 Auteur
Ton Nom — Kaoutar MEZOUAHI en 4ème année Ingénierie AI & ML
Projet réalisé dans le cadre du module NLP & RAG

📚 Ressources utilisées
LangChain
Ollama
FAISS
Streamlit
Sentence Transformers (all-MiniLM-L6-v2)


---

## 🧠 Mémoire conversationnelle (Contexte multi-tours)

Modifie `src/rag_chain.py` pour ajouter la mémoire :

```python
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.vector_store import load_vector_store
import yaml

CONDENSE_PROMPT = """Étant donné l'historique de conversation suivant et une question de suivi, reformulez la question de suivi pour qu'elle soit autonome.

Historique :
{chat_history}

Question de suivi : {question}

Question reformulée :"""

QA_PROMPT = """Tu es un assistant médical intelligent. Tu réponds UNIQUEMENT à partir du contexte médical fourni.
Si tu ne trouves pas la réponse dans le contexte, dis "Je ne dispose pas d'informations suffisantes dans ma base de connaissances pour répondre à cette question."

Contexte médical :
{context}

Question : {question}

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
    
    # Mémoire conversationnelle
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Chaîne avec mémoire
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": config['rag']['top_k']}),
        memory=memory,
        condense_question_prompt=PromptTemplate(
            template=CONDENSE_PROMPT,
            input_variables=["chat_history", "question"]
        ),
        combine_docs_chain_kwargs={"prompt": PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "question"]
        )},
        return_source_documents=True,
        verbose=False
    )
    
    return qa_chain
