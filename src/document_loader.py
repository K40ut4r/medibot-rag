# src/document_loader.py
import re
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader

def extract_section_from_page(text: str, page_num: int) -> str:
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and (line.isupper() or re.match(r'^\d+[\.\s]', line)) and len(line) < 100:
            return line
    return f"Section_{page_num}"

def compute_reliability(pdf_name: str) -> float:
    """Attribue un score de fiabilité selon la source du document."""
    name_upper = pdf_name.upper()
    if "HAS" in name_upper:
        return 0.98
    elif "WHO" in name_upper or "OMS" in name_upper:
        return 0.95
    elif "ANSM" in name_upper:
        return 0.90
    else:
        return 0.85

def load_pdfs(pdf_dir: str) -> List:
    documents = []
    pdf_path = Path(pdf_dir)
    
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"   📖 Lecture de {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        reliability = compute_reliability(pdf_file.name)
        
        for i, doc in enumerate(docs):
            doc.metadata["source"] = pdf_file.name
            doc.metadata["type"] = "medical_guideline"
            doc.metadata["section"] = extract_section_from_page(doc.page_content, i)
            doc.metadata["reliability_score"] = reliability  # ✅ Hérité par tous les chunks
            doc.metadata["page"] = i + 1
        
        documents.extend(docs)
    
    return documents# src/document_loader.py
import re
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader

def extract_section_from_page(text: str, page_num: int) -> str:
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and (line.isupper() or re.match(r'^\d+[\.\s]', line)) and len(line) < 100:
            return line
    return f"Section_{page_num}"

def compute_reliability(pdf_name: str) -> float:
    name_upper = pdf_name.upper()
    if "HAS" in name_upper:
        return 0.98
    elif "FCM" in name_upper:
        return 0.95
    elif "ACS" in name_upper or "CANCER.ORG" in name_upper:
        return 0.95  # American Cancer Society
    elif "WHO" in name_upper or "OMS" in name_upper:
        return 0.95
    elif "MOH" in name_upper or "MINISTRY" in name_upper:
        return 0.88
    elif any(x in name_upper for x in ["GUIDELINE", "RECOMMENDATION", "2024", "2025"]):
        return 0.90
    else:
        return 0.85

def load_pdfs(pdf_dir: str) -> List:
    documents = []
    pdf_path = Path(pdf_dir)
    
    for pdf_file in pdf_path.glob("*.pdf"):
        print(f"   📖 Lecture de {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        reliability = compute_reliability(pdf_file.name)
        
        for i, doc in enumerate(docs):
            doc.metadata["source"] = pdf_file.name
            doc.metadata["type"] = "medical_guideline"
            doc.metadata["section"] = extract_section_from_page(doc.page_content, i)
            doc.metadata["reliability_score"] = reliability  # ✅ Hérité par tous les chunks
            doc.metadata["page"] = i + 1
        
        documents.extend(docs)
    
    return documents