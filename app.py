import streamlit as st
from src.rag_chain import build_rag_chain

st.set_page_config(page_title="MediBot RAG", page_icon="🩺", layout="wide")

st.title("🩺 MediBot — Assistant Médical RAG")
st.caption("Propulsé par LangChain + Ollama | Sources vérifiées | 💬 Mémoire conversationnelle")

st.warning("⚕️ **Avertissement** : Cet outil est à titre informatif uniquement. Consultez un médecin pour tout diagnostic.")

# ─── INITIALISATION ───
if "chain" not in st.session_state:
    with st.spinner("🚀 Chargement du modèle RAG..."):
        st.session_state.chain = build_rag_chain()
        st.session_state.messages = []
        st.session_state.processing = False

# ─── SIDEBAR ───
with st.sidebar:
    st.header("⚙️ Informations")
    st.info("💡 Le bot garde le contexte de la conversation.")
    
    if st.button("🗑️ Effacer l'historique", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processing = False
        st.rerun()
    
    st.divider()
    st.markdown("📥 **Télécharger le résumé**")
    if st.session_state.messages:
        conversation = "\n\n".join([
            f"{'👤 Patient' if m['role']=='user' else '🤖 MediBot'}: {m['content']}" 
            for m in st.session_state.messages
        ])
        st.download_button(
            label="📄 Exporter la conversation",
            data=conversation,
            file_name="consultation_medibot.txt",
            mime="text/plain",
            use_container_width=True
        )

# ─── FONCTION DE REFORMULATION (MÉMOIRE) ───
def reformulate_question(question: str, chat_history: list) -> str:
    if not chat_history or len(chat_history) < 2:
        return question
    
    # Trouve le DERNIER sujet médical distinct (pas la dernière question)
    medical_keywords = ["diabète", "obésité", "helicobacter", "pylori", "cancer", "asthme", 
                        "grippe", "vaccin", "hypertension", "cholestérol", "ulcère", "gastrite"]
    
    last_subject = None
    # Parcourt l'historique à l'envers pour trouver le dernier sujet
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            content = msg["content"].lower()
            for kw in medical_keywords:
                if kw in content:
                    last_subject = content
                    break
            if last_subject:
                break
    
    if not last_subject:
        return question
    
    # Si la question est courte/vague, reformule avec le dernier sujet
    followup_indicators = ["et ", "quels ", "quel ", "comment ", "pourquoi ", "où ", "quand ", 
                           "qui ", "les ", "le ", "la ", "des ", "du ", "de ", "ce ", "cette ", "cet "]
    is_followup = len(question) < 80 or any(question.lower().startswith(ind) for ind in followup_indicators)
    
    # Vérifie si la question contient déjà un sujet médical
    has_subject = any(kw in question.lower() for kw in medical_keywords)
    
    if is_followup and not has_subject:
        return f"Concernant {last_subject}, {question}"
    
    return question

# ─── AFFICHAGE HISTORIQUE ───
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources utilisées"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# ─── BLOCAGE INPUT SI TRAITEMENT EN COURS ───
if st.session_state.processing:
    st.chat_input("⏳ Le bot répond... Veuillez patienter.", disabled=True)
else:
    if prompt := st.chat_input("Décrivez vos symptômes..."):
        
        # 🚨 GUARDRAIL : Détection d'urgence
        URGENCY_KEYWORDS = [
            "crise cardiaque", "arrêt cardiaque", "saignement", "perte de connaissance",
            "difficulté à respirer", "étouffement", "empoisonnement", "suicide",
            "je ne peux pas respirer", "je meurs", "urgence vitale"
        ]
        is_urgency = any(kw in prompt.lower() for kw in URGENCY_KEYWORDS)
        
        # Marque comme "en traitement"
        st.session_state.processing = True
        
        # Ajoute le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# ─── TRAITEMENT DE LA DERNIÈRE QUESTION ───
if st.session_state.processing and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        
        # Vérifie urgence
        URGENCY_KEYWORDS = [
            "crise cardiaque", "arrêt cardiaque", "saignement", "perte de connaissance",
            "difficulté à respirer", "étouffement", "empoisonnement", "suicide",
            "je ne peux pas respirer", "je meurs", "urgence vitale"
        ]
        is_urgency = any(kw in prompt.lower() for kw in URGENCY_KEYWORDS)
        
        if is_urgency:
            answer = "🚨 **Cette situation nécessite une intervention médicale IMMÉDIATE.**\n\nAppelez le **15 (SAMU)** ou le **112**."
            sources = []
            confidence = 1.0
            st.error(answer)
            
        else:
            with st.spinner("🔍 Recherche dans la base médicale..."):
                
                # ✅ REFORMULATION AVEC MÉMOIRE
                standalone_question = reformulate_question(prompt, st.session_state.messages[:-1])
                if standalone_question != prompt:
                    st.caption(f"🔄 Question reformulée : *{standalone_question}*")
                
                result = st.session_state.chain.invoke(standalone_question)
                docs = result.get("source_documents", [])
                sources = list(set([doc.metadata.get("source", "Inconnu") for doc in docs]))
                
                if not docs:
                    answer = "Je ne dispose pas d'informations suffisantes dans ma base de connaissances pour répondre à cette question. Consultez un professionnel de santé."
                    confidence = 0.2
                    st.info(answer)
                else:
                    answer = result["answer"]
                    
                    # Nettoyage fallback
                    fallback_phrases = [
                        "Je ne dispose pas d'informations",
                        "Aucun document pertinent trouvé",
                        "Consultez un professionnel de santé pour obtenir des conseils adéquats"
                    ]
                    for phrase in fallback_phrases:
                        if phrase in answer and len(answer) > len(phrase) + 50:
                            answer = answer.replace(phrase, "").strip()
                            answer = answer.replace("..", ".").replace("  ", " ")
                    
                    if len(sources) >= 3:
                        confidence = 0.95
                    elif len(sources) == 2:
                        confidence = 0.85
                    elif len(sources) == 1:
                        confidence = 0.70
                    else:
                        confidence = 0.40
                    
                    st.markdown(answer)
        
        st.progress(min(confidence, 1.0), text=f"Fiabilité des sources: {confidence*100:.0f}%")
        
        if sources:
            with st.expander("📚 Sources utilisées"):
                for s in sources:
                    st.markdown(f"- `{s}`")
    
    # Sauvegarde + libère le flag
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state.processing = False
    st.rerun()