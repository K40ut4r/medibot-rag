# 🩺 MediBot RAG — Assistant Médical Intelligent

&gt; Projet de fin de module NLP & RAG — 4ème année Ingénierie AI/ML

MediBot est un agent conversationnel médical basé sur **RAG (Retrieval-Augmented Generation)** qui répond aux questions de santé en s'appuyant sur des sources scientifiques fiables (HAS, WHO). Contrairement aux LLM classiques, MediBot **cite toujours ses sources** et inclut des avertissements médicaux.

## 🎯 Fonctionnalités

- ✅ **RAG médical** : Réponses basées sur des PDFs médicaux officiels
- ✅ **Citations de sources** : Chaque réponse indique le document PDF source
- ✅ **Garde-fous sécurité & Intentions** : Détection d'urgence + orientation pour prise de rdv
- ✅ **Multilingue** : Détecte et répond en Français, Anglais ou Arabe
- ✅ **LLM local** (Ollama) — 100% offline, confidentialité des données

## 🚀 Installation rapide

```bash
# 1. Environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Mac/Linux

# 2. Dépendances
pip install -r requirements.txt

# 3. Télécharger un modèle Ollama (léger, rapide)
ollama pull phi3:mini

# 4. Mettre vos PDFs médicaux dans data/raw/

# 5. Créer la base vectorielle
python ingest.py

# 6. Lancer l'application
streamlit run app.py

L'application sera accessible sur http://localhost:8501
📁 Structure du projet
medibot-rag/
├── app.py                    # Interface Streamlit
├── ingest.py                 # Script d'ingestion des PDFs
├── config.yaml               # Configuration (modèle, chunks, etc.)
├── requirements.txt
├── evaluation_results.json   # Résultats des évaluations du modèle
├── README.md
├── notebooks/
│   └── 01_ingestion.ipynb    # Tests et expérimentations d'ingestion
├── src/
│   ├── __init__.py
│   ├── document_loader.py    # Extraction texte PDF
│   ├── chunker.py            # Découpage en chunks
│   ├── vector_store.py       # Base FAISS + embeddings
│   ├── rag_chain.py          # Chaîne RAG complète
│   ├── agent_medibot.py      # Agent intelligent (mémoire et routing)
│   ├── intent_detector.py    # Détection d'intentions médicales/urgences
│   ├── evaluation.py         # Pipeline d'évaluation (RAGAS / LLM-as-a-judge)
│   ├── evaluator.py          # Logique d'évaluation
│   └── utils.py              # Fonctions utilitaires
└── data/
    ├── raw/                  # PDFs source (non versionnés)
    └── faiss_index/          # Base vectorielle générée

    🧪 Exemple d'utilisation
Question : "Quels sont les symptômes du diabète de type 2 ?"
Réponse :
Les symptômes du diabète de type 2 peuvent inclure une forte soif, la polyurie, la fatigue, des plaies lentes à guérir...
📚 Source : 
parcours_de_soins_du_patient_adulte_vivant_avec_un_diabete.pdf

Question de suivi : "Et quels sont les risques ?"
Réponse :
Le diabète de type 2 augmente le risque de maladies cardiovasculaires, de problèmes renaux...
📚 Source : parcours_de_soins_du_patient_adulte_vivant_avec_un_diabete.pdf
🛡️ Avertissement
Ce projet est à but éducatif uniquement. Les informations fournies ne remplacent en aucun cas un avis médical professionnel. En cas d'urgence, contactez le 15 (SAMU) ou le 112.
👨‍💻 Auteur
MEZOUAHI Kaoutar — Étudiante en 4ème année Ingénierie AI ML
Projet réalisé dans le cadre du module NLP & RAG


---

