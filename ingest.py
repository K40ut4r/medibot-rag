from src.document_loader import load_pdfs
from src.chunker import split_documents
from src.vector_store import create_vector_store
import yaml

def main():
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("📄 Chargement des PDFs...")
    docs = load_pdfs(config['paths']['pdf_dir'])
    print(f"✅ {len(docs)} pages chargées")
    
    print("✂️ Chunking...")
    chunks = split_documents(docs, config['rag']['chunk_size'], config['rag']['chunk_overlap'])
    print(f"✅ {len(chunks)} chunks créés")
    
    print("🧠 Création de la base vectorielle FAISS...")
    create_vector_store(chunks, config['paths']['faiss_index'], config['rag']['model_name'])
    print("✅ Ingestion terminée !")

if __name__ == "__main__":
    main()