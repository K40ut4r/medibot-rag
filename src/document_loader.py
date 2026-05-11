from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(pdf_dir: str) -> List:
    """Charge tous les PDFs d'un dossier et retourne une liste de documents."""
    documents = []
    pdf_path = Path(pdf_dir)
    
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"   📖 Lecture de {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["type"] = "medical_guideline"
        
        documents.extend(docs)
    
    return documents