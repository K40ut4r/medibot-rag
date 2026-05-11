# medibot-rag
medibot-rag/
├── README.md                 # Description + instructions d'installation
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py                    # Streamlit UI
├── config.yaml               # Hyperparamètres du RAG
├── data/
│   ├── raw/                  # Mets tes PDFs ici (pas sur GitHub !)
│   └── chroma_db/            # Base vectorielle (générée, gitignorée)
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Extraction texte PDF
│   ├── chunker.py            # Découpage intelligent
│   ├── vector_store.py       # ChromaDB + embeddings
│   ├── rag_chain.py          # Chaîne LangChain complète
│   ├── evaluator.py          # Métriques d'évaluation
│   └── utils.py              # Helpers
└── notebooks/
    └── 01_ingestion.ipynb    # Pour tester l'ingestion
