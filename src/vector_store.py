from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import os
import torch

def create_vector_store(documents: List[Document], persist_dir: str, model_name: str):
    os.makedirs(persist_dir, exist_ok=True)
    
    print(f"   🔧 Chargement du modèle d'embeddings {model_name}...")
    
    # Forcer CPU + éviter le bug meta tensor
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': 'cpu',
            'trust_remote_code': False
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'device': 'cpu'
        }
    )
    
    print("   📊 Création des vecteurs...")
    vectordb = FAISS.from_documents(documents, embeddings)
    
    print(f"   💾 Sauvegarde dans {persist_dir}...")
    vectordb.save_local(persist_dir)
    return vectordb

def load_vector_store(persist_dir: str, model_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': 'cpu',
            'trust_remote_code': False
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'device': 'cpu'
        }
    )
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)