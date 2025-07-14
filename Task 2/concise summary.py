import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

if 'summary_count' not in st.session_state:
    st.session_state.summary_count = 0
    st.session_state.common_words = Counter()
    st.session_state.user_ratings = []

def get_filtered_words(text):
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    words = word_tokenize(text.lower())
    return [word for word in words if word not in stop_words and word.isalnum()]

def main():
    st.set_page_config(page_title="Summary Generator with Analytics", layout="wide")
    st.title("📝 Concise Summary Generator")

    summary_tab, analytics_tab = st.tabs(["📄 Summary Generator", "📈 Analytics"])

    with summary_tab:
        input_mode = st.radio("Choose input method:", ["Enter Text", "Upload Document"])

        if input_mode == "Enter Text":
            user_input = st.text_area("Enter your text here:", height=300)
        else:
            uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
            if uploaded_file:
                user_input = uploaded_file.read().decode("utf-8")
            else:
                user_input = ""

        num_sentences = st.slider("Number of summary sentences:", min_value=1, max_value=10, value=3)

        if st.button("Generate Summary"):
            if user_input.strip():
                user_input = ' '.join(user_input.split())
                sentences = sent_tokenize(user_input)
                filtered_words = get_filtered_words(user_input)
                word_freq = FreqDist(filtered_words)
                st.session_state.summary_count += 1
                st.session_state.common_words.update(filtered_words)
                sentence_scores = {}
                for sentence in sentences:
                    words_in_sentence = word_tokenize(sentence.lower())
                    score = sum(word_freq.get(word, 0) for word in words_in_sentence)
                    sentence_scores[sentence] = score
                top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
                top_sentences = sorted(top_sentences, key=sentences.index)
                st.subheader("🧾 Summary:")
                st.write(' '.join(top_sentences))
                st.subheader("📊 Rate this summary")
                rating = st.radio("How useful was this summary?", [1, 2, 3, 4, 5], horizontal=True)
                if st.button("Submit Rating"):
                    st.session_state.user_ratings.append(rating)
                    st.success("Thanks for your feedback! 🎉")
            else:
                st.warning("Please enter text or upload a file first.")

    with analytics_tab:
        st.header("📈 Summary Usage Analytics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Summaries Generated", st.session_state.summary_count)
        with col2:
            if st.session_state.user_ratings:
                avg = round(sum(st.session_state.user_ratings) / len(st.session_state.user_ratings), 2)
            else:
                avg = "N/A"
            st.metric("Avg. Satisfaction Rating", avg)
        with col3:
            word_count = sum(st.session_state.common_words.values())
            st.metric("Words Processed", word_count)
        st.subheader("🔤 Most Common Words")
        if st.session_state.common_words:
            top_words = st.session_state.common_words.most_common(10)
            words, freqs = zip(*top_words)
            st.bar_chart({w: f for w, f in top_words})
        else:
            st.info("No words processed yet. Generate a summary to see stats.")

if __name__ == "__main__":
    main()
