import streamlit as st
import pandas as pd
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_data
def load_medquad_from_csv(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        if 'Answer' not in df.columns:
            st.error("❌ 'Answer' column not found.")
            return []

        qa_pairs = []
        for text in df['Answer'].dropna():
            q_match = re.search(r'Question:\s*(.*?)\s*URL:', text, re.DOTALL)
            a_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)

            if q_match and a_match:
                question = q_match.group(1).strip().replace('\n', ' ')
                answer = a_match.group(1).strip().replace('\n', ' ')
                qa_pairs.append((question, answer))

        if not qa_pairs:
            st.warning("⚠️ No valid Q&A found.")
        else:
            st.success(f"✅ Loaded {len(qa_pairs)} Q&A pairs.")
        return qa_pairs

    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        return []

def preprocess(text):
    return re.sub(f"[{string.punctuation}]", " ", text.lower())

@st.cache_data
def build_index(pairs):
    questions = [preprocess(q) for q, _ in pairs]
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(questions)
    return tfidf, matrix

def retrieve_answer(query, tfidf, matrix, pairs):
    q_vec = tfidf.transform([preprocess(query)])
    scores = cosine_similarity(q_vec, matrix).flatten()
    best = np.argmax(scores)
    return pairs[best]

def extract_entities(text):
    symptoms = ['fever','cough','pain','fatigue','nausea','headache']
    diseases = ['diabetes','cancer','asthma','covid','flu','pcos','noonan','obesity']
    treatments = ['surgery','medication','therapy','vaccine','genetic counseling']
    t = text.lower()
    return {
        "Symptoms": [s for s in symptoms if s in t],
        "Diseases": [d for d in diseases if d in t],
        "Treatments": [x for x in treatments if x in t]
    }

def main():
    st.set_page_config(page_title="Medical Q&A Chatbot", layout="wide")
    st.title("🩺 Medical Q&A Chatbot (MedQuAD CSV)")
    st.write("Ask a medical question and get an answer based on the MedQuAD dataset.")

    file_path = "All-2479-Answers-retrieved-from-MedQuAD.csv"
    qa_pairs = load_medquad_from_csv(file_path)
    if not qa_pairs:
        return

    tfidf, matrix = build_index(qa_pairs)

    user_query = st.text_input("Ask your medical question:")
    if user_query:
        matched_q, answer = retrieve_answer(user_query, tfidf, matrix, qa_pairs)
        entities = extract_entities(user_query)

        st.subheader("🔍 Closest Match")
        st.markdown(f"**{matched_q}**")

        st.subheader("💬 Answer")
        st.write(answer)

        st.subheader("🧬 Medical Entities")
        for label, items in entities.items():
            st.write(f"**{label}:** {', '.join(items) if items else 'None'}")

if __name__ == "__main__":
    main()
