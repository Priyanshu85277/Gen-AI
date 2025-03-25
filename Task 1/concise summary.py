import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

nltk.download('punkt')
nltk.download('stopwords')

def main():
    st.title("Concise Summary Generator")

    input_choice = st.radio("Input:", ("Enter Text", "Upload Document"))

    if input_choice == "Enter Text":
        input_text = st.text_area("Enter the text:", height=300)
    else:
        uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
        if uploaded_file is not None:
            input_text = uploaded_file.read().decode("utf-8")
        else:
            input_text = ""

    num_sentences = st.slider("Select the number of sentences:", min_value=1, max_value=10, value=5)

    if st.button("Generate"):
        if input_text.strip():
            sentences = sent_tokenize(input_text)
            stop_words = set(stopwords.words('english') + list(string.punctuation))
            words = word_tokenize(input_text.lower())
            filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
            word_frequencies = FreqDist(filtered_words)
            
            sentence_scores = {}
            for sentence in sentences:
                sentence_words = word_tokenize(sentence.lower())
                sentence_score = sum(word_frequencies[word] for word in sentence_words if word in word_frequencies)
                sentence_scores[sentence] = sentence_score
            
            summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
            st.subheader("Concise Summary:")
            st.write(' '.join(summary_sentences))
        else:
            st.warning("Set input text or upload Document.")

if __name__ == "__main__":
    main()
