import streamlit as st
import pandas as pd
import os
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_medquad_from_csv(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        if 'Answer' not in df.columns:
            st.warning(f"⚠️ Skipping {filepath} — missing 'Answer' column.")
            return []

        qa_pairs = []
        for text in df['Answer'].dropna():
            q_match = re.search(r'Question:\s*(.*?)\s*URL:', text, re.DOTALL)
            a_match = re.search(r'Answer:\s*(.*)', text, re.DOTALL)

            if q_match and a_match:
                question = q_match.group(1).strip().replace('\n', ' ')
                answer = a_match.group(1).strip().replace('\n', ' ')
                qa_pairs.append((question, answer))

        return qa_pairs
    except Exception as e:
        st.error(f"❌ Error reading {filepath}: {e}")
        return []

@st.cache_data
def load_all_medquad_csvs(folder="data"):
    all_pairs = []
    for fname in os.listdir(folder):
        if fname.endswith(".csv"):
            fpath = os.path.join(folder, fname)
            pairs = load_medquad_from_csv(fpath)
            all_pairs.extend(pairs)
    unique_pairs = list({q: (q, a) for q, a in all_pairs}.values())
    st.success(f"✅ Loaded {len(unique_pairs)} Q&A pairs from `{folder}/`")
    return unique_pairs

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
    return pairs[best], scores[best]

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
    st.title("🩺 Medical Q&A Chatbot")
    st.caption("Ask medical questions. The bot uses all CSV files inside the `data/` folder.")

    if st.button("🔄 Refresh Knowledge Base"):
        st.cache_data.clear()
        st.experimental_rerun()

    qa_pairs = load_all_medquad_csvs("data")
    if not qa_pairs:
        st.error("No Q&A pairs found.")
        return

    tfidf, matrix = build_index(qa_pairs)

    user_query = st.text_input("Ask a medical question:")
    if user_query:
        (matched_q, answer), score = retrieve_answer(user_query, tfidf, matrix, qa_pairs)
        entities = extract_entities(user_query)

        st.markdown("---")
        st.subheader("Closest Matched Question")
        st.markdown(f"**{matched_q}**")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Similarity Score")
        st.write(f"{score:.4f}")

        st.subheader("Extracted Medical Entities")
        for label, items in entities.items():
            st.markdown(f"**{label}:** {', '.join(items) if items else 'None'}")

if __name__ == "__main__":
    main()
