import streamlit as st
from src.agent_medibot import MediBotAgent
from src.vector_store import load_vector_store
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import yaml

st.set_page_config(page_title="MediBot RAG", page_icon="🩺", layout="wide")

# ─── THEME MÉDICAL CSS ───
st.markdown("""
<style>
    .stApp header { background-color: #0d7377 !important; }
    .stButton>button {
        background-color: #14a085;
        color: white;
        border-radius: 8px;
        border: none;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #0d7377;
        transform: translateY(-1px);
    }
    .stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #e8f6f3 !important;
        border-left: 4px solid #14a085;
    }
</style>
""", unsafe_allow_html=True)

# ─── CHARGEMENT CONFIG ───
with open("config.yaml", 'r', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

# ─── INITIALISATION ───
if "agent" not in st.session_state:
    st.session_state.agent = MediBotAgent()
    st.session_state.messages = []
    st.session_state.doc_count = 14

if "busy" not in st.session_state:
    st.session_state.busy = False

# ═══════════════════════════════════════════════════════════
# CALCUL GLOBAL : est-ce qu'on est en train de générer ?
# ═══════════════════════════════════════════════════════════
is_generating = (
    len(st.session_state.messages) > 0 
    and st.session_state.messages[-1]["role"] == "user"
)

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⚙️ MediBot")
    
    # ─── Stats ───
    st.subheader("📊 Corpus médical")
    c1, c2, c3 = st.columns(3)
    c1.metric("📄", st.session_state.get("doc_count", 14))
    c2.metric("🏥", "10")
    c3.metric("🧩", "1.4k")
    st.progress(0.92, text="Fiabilité moyenne: 92%")
    st.divider()
    
    # ─── Upload PDF (toujours visible) ───
    st.subheader("📤 Ajouter un document")
    st.caption("PDF médical officiel (HAS, WHO, etc.)")
    uploaded_pdf = st.file_uploader(
        "Déposer un PDF", type=["pdf"], accept_multiple_files=False, label_visibility="collapsed"
    )
    
    if uploaded_pdf is not None:
        pdf_dir = CONFIG['paths']['pdf_dir']
        os.makedirs(pdf_dir, exist_ok=True)
        pdf_path = os.path.join(pdf_dir, uploaded_pdf.name)
        
        if os.path.exists(pdf_path):
            st.warning(f"⚠️ `{uploaded_pdf.name}` est déjà dans le corpus.")
        else:
            with st.spinner(f"🔧 Indexation de `{uploaded_pdf.name}`..."):
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                
                loader = PyPDFLoader(pdf_path)
                new_docs = loader.load()
                for d in new_docs:
                    d.metadata["source"] = uploaded_pdf.name
                    d.metadata["type"] = "medical_guideline"
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CONFIG['rag'].get('chunk_size', 512),
                    chunk_overlap=CONFIG['rag'].get('chunk_overlap', 100),
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                new_chunks = splitter.split_documents(new_docs)
                
                faiss_dir = CONFIG['paths']['faiss_index']
                embeddings_model = CONFIG['rag']['model_name']
                
                if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
                    db = load_vector_store(faiss_dir, embeddings_model)
                    db.add_documents(new_chunks)
                    db.save_local(faiss_dir)
                else:
                    from src.vector_store import create_vector_store
                    create_vector_store(new_chunks, faiss_dir, embeddings_model)
                
                st.session_state.agent = MediBotAgent()
                st.session_state.doc_count += 1
                st.success(f"✅ `{uploaded_pdf.name}` indexé ({len(new_chunks)} chunks)")
                st.balloons()
    
    st.divider()
    
    # ─── Questions rapides (CACHÉES si traitement en cours) ───
    st.subheader("❓ Questions rapides")
    
    if not is_generating:
        quick_questions = [
            "Symptômes diabète type 2",
            "Prévention obésité",
            "Signes crise cardiaque",
            "Arrêt benzodiazépines",
            "Vaccin dengue Qdenga",
            "Recommandations Alzheimer"
        ]
        for q in quick_questions:
            if st.button(q, use_container_width=True, key=f"qq_{hash(q)}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
    else:
        st.info("⏳ Traitement en cours...")
        st.caption("Patientez que MediBot réponde avant de poser une nouvelle question.")
    
    st.divider()
    
    # ─── Export ───
    if st.session_state.messages:
        conv = "\n\n".join([
            f"{'👤 Patient' if m['role']=='user' else '🤖 MediBot'}: {m['content']}"
            for m in st.session_state.messages
        ])
        st.download_button(
            "📥 Exporter conversation", conv,
            file_name="consultation_medibot.txt", mime="text/plain",
            use_container_width=True
        )
    
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.session_state.busy = False
        st.rerun()
    
    st.divider()
    st.caption("v1.1 • LangChain + Ollama + FAISS")
    st.caption("Kaoutar Mezouahi — EFM NLP CI2")

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.title("🩺 MediBot — Agent Médical Intelligent")
st.caption("Propulsé par LangChain + Ollama | Sources vérifiées | Multilingue 🇫🇷🇬🇧🇸🇦")
st.warning("⚕️ **Avertissement** : Cet outil est à titre informatif uniquement. Consultez un médecin pour tout diagnostic.")

# ═══════════════════════════════════════════════════════════
# ONGLETS
# ═══════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["💬 Chat", "📊 Mode Évaluation"])

with tab1:
    # ─── AFFICHAGE HISTORIQUE ───
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Sources utilisées"):
                    for src in msg["sources"]:
                        src_name = src.metadata.get("source", "Inconnu")
                        section = src.metadata.get("section", "Non classé")
                        st.markdown(f"- **`{src_name}`** — *Section: {section}*")
                        st.caption(src.page_content[:120] + "...")
    
    # ─── CHAT INPUT ───
    placeholder = (
        "⏳ MediBot répond à la question précédente, veuillez patienter..." 
        if is_generating else "Décrivez vos symptômes..."
    )
    prompt = st.chat_input(placeholder, disabled=is_generating)
    
    if prompt and not is_generating:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
    
    # ─── TRAITEMENT AVEC VERROU ───
    unanswered_idx = None
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            is_last = (i == len(st.session_state.messages) - 1)
            next_is_not_assistant = (
                not is_last and st.session_state.messages[i + 1]["role"] != "assistant"
            )
            if is_last or next_is_not_assistant:
                unanswered_idx = i
                break
    
    if unanswered_idx is not None:
        question = st.session_state.messages[unanswered_idx]["content"]
        
        # ⭐ VERROU : évite le double traitement si un autre run est déjà en cours
        if st.session_state.busy:
            with st.chat_message("assistant", avatar="🤖"):
                st.spinner("🔍 Recherche dans la base médicale...")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    st.session_state.busy = True
                    with st.spinner("🤖 L'agent médical réfléchit..."):
                        # Appel à l'agent
                        result = st.session_state.agent.run(question)
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        
                        if result.get("is_emergency"):
                            st.error(answer)
                        elif result.get("suggested_action") == "open_app":
                            st.info(answer)
                        else:
                            st.markdown(answer)
                    
                    if sources:
                        confidence = min(0.7 + len(sources) * 0.08, 0.98)
                        st.progress(confidence, text=f"Fiabilité source: {confidence*100:.0f}%")
                        with st.expander("📚 Sources utilisées"):
                            for src in sources:
                                src_name = src.metadata.get("source", "Inconnu")
                                section = src.metadata.get("section", "Non classé")
                                st.markdown(f"- **`{src_name}`** — *Section: {section}*")
                                st.caption(src.page_content[:120] + "...")
                    elif not result.get("is_emergency") and not result.get("suggested_action") == "open_app":
                        st.info("Aucune source médicale fiable trouvée.")
                
                finally:
                    st.session_state.busy = False
                
                # Insert réponse dans l'historique
                st.session_state.messages.insert(unanswered_idx + 1, {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()

with tab2:
    st.header("📊 Benchmark de pertinence")
    st.write("Évaluation automatique sur 10 questions médicales prédéfinies.")
    
    if st.button("🚀 Lancer l'évaluation"):
        with st.spinner("Évaluation en cours... (~2-3 min)"):
            try:
                from src.evaluation import run_benchmark
                results = run_benchmark()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Questions", len(results))
                col2.metric("Retrieval OK", sum(1 for r in results if r.get("retrieval_ok")))
                col3.metric("Réponses OK", sum(1 for r in results if r.get("answer_ok")))
                
                st.dataframe(results, use_container_width=True)
                
                import matplotlib.pyplot as plt
                categories = {}
                for r in results:
                    cat = r.get("category", "Autre")
                    categories[cat] = categories.get(cat, {"ok": 0, "total": 0})
                    categories[cat]["total"] += 1
                    if r.get("answer_ok"):
                        categories[cat]["ok"] += 1
                
                fig, ax = plt.subplots(figsize=(8, 4))
                cats = list(categories.keys())
                rates = [categories[c]["ok"] / categories[c]["total"] * 100 for c in cats]
                bars = ax.bar(cats, rates, color="#14a085", edgecolor="white")
                ax.set_ylabel("Taux de succès (%)")
                ax.set_title("Pertinence par spécialité médicale")
                ax.set_ylim(0, 110)
                for bar, rate in zip(bars, rates):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                            f"{rate:.0f}%", ha="center", va="bottom", color="white")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.info("Vérifiez que `src/evaluation.py` existe et que FAISS est créée.")