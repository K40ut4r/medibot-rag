import streamlit as st
from src.rag_chain import build_rag_chain

st.set_page_config(page_title="MediBot RAG", page_icon="🩺", layout="wide")

st.title("🩺 MediBot — Assistant Médical RAG")
st.caption("Propulsé par LangChain + Ollama | Sources vérifiées")

st.warning("⚕️ **Avertissement** : Cet outil est à titre informatif uniquement. Consultez un médecin pour tout diagnostic.")

# Initialisation — ne se fait qu'une seule fois
if "chain" not in st.session_state:
    with st.spinner("🚀 Chargement du modèle RAG..."):
        st.session_state.chain = build_rag_chain()
        st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Informations")
    st.info("💡 Le bot répond à partir de documents médicaux PDF.")
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# Affichage historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src}")

# Input
if prompt := st.chat_input("Décrivez vos symptômes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Recherche dans la base médicale..."):
            # ✅ Utilise .invoke() avec "query" (pas "question")
            result = st.session_state.chain.invoke({"query": prompt})
            answer = result["result"]
            sources = list(set([doc.metadata.get("source", "Inconnu") for doc in result["source_documents"]]))
        
        st.markdown(answer)
        with st.expander("📚 Sources utilisées"):
            for s in sources:
                st.markdown(f"- `{s}`")
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })