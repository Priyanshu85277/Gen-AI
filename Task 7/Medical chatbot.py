import streamlit as st
import os
import base64
from PIL import Image
import requests
from io import BytesIO
from textblob import TextBlob
from dotenv import load_dotenv
from langdetect import detect
from deep_translator import GoogleTranslator

load_dotenv()
GOOGLE_API_KEY = os.getenv("AIzaSyA0JpFuXyv64fViTZIXRVsnqOoyuzPUKBs")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": GOOGLE_API_KEY
}

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French"
}

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "en"
    except:
        return "en"

def translate_text(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except:
        return text

def query_gemini(text_input, image_data=None):
    contents = [{"parts": [{"text": text_input}]}]
    if image_data:
        contents[0]["parts"].append({
            "inlineData": {
                "mimeType": "image/jpeg",
                "data": image_data
            }
        })
    payload = {"contents": contents}
    try:
        res = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        st.error(f"Gemini API Error: {res.status_code} - {res.text}")
        return "❌ Error processing the input."

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_empathy_prefix(sentiment, lang_code):
    messages = {
        "positive": {
            "en": "😊 I'm glad to hear that! Here's some helpful information:\n\n",
            "hi": "😊 यह सुनकर अच्छा लगा! यहाँ कुछ उपयोगी जानकारी है:\n\n",
            "es": "😊 ¡Me alegra escuchar eso! Aquí hay información útil:\n\n",
            "fr": "😊 Je suis content d'entendre cela ! Voici quelques informations utiles :\n\n",
        },
        "negative": {
            "en": "😟 I'm sorry you're experiencing this. Let me help:\n\n",
            "hi": "😟 मुझे खेद है कि आप ऐसा महसूस कर रहे हैं। मैं मदद करता हूँ:\n\n",
            "es": "😟 Lo siento mucho. Déjame ayudarte:\n\n",
            "fr": "😟 Je suis désolé que vous ressentiez cela. Laissez-moi vous aider :\n\n",
        },
        "neutral": {
            "en": "🙂 Here's what I found based on your input:\n\n",
            "hi": "🙂 आपके विवरण के आधार पर जानकारी:\n\n",
            "es": "🙂 Esto es lo que encontré según tu entrada:\n\n",
            "fr": "🙂 Voici ce que j'ai trouvé en fonction de votre question :\n\n",
        }
    }
    return messages.get(sentiment, {}).get(lang_code, messages["neutral"]["en"])

def main():
    st.set_page_config(page_title="🩺 Multilingual Medical Chatbot", layout="wide")
    st.title("🌍 Gemini-Powered Medical Chatbot (Multilingual + Emotion Aware)")
    st.caption("Auto-detects language, adjusts tone, and responds accordingly.")

    with st.form("med_form"):
        user_query = st.text_area("💬 Ask your question in any language:")
        image_file = st.file_uploader("🖼 Upload medical image (optional)", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Submit")

    if submitted and user_query:
        detected_lang = detect_language(user_query)
        translated_query = translate_text(user_query, detected_lang, "en")
        sentiment = analyze_sentiment(translated_query)
        empathy_prefix = generate_empathy_prefix(sentiment, detected_lang)
        img_data = encode_image(image_file) if image_file else None

        with st.spinner("🧠 Processing with Gemini..."):
            gemini_response_en = query_gemini(translated_query, img_data)
            gemini_response_local = translate_text(gemini_response_en, "en", detected_lang)
            final_response = empathy_prefix + gemini_response_local

        emoji_map = {"positive": "😊", "neutral": "😐", "negative": "😢"}
        st.subheader(f"🔍 Detected Language: `{SUPPORTED_LANGUAGES.get(detected_lang, 'English')}`")
        st.markdown(f"**Sentiment:** {emoji_map[sentiment]} `{sentiment.capitalize()}`")
        st.subheader("💬 Gemini's Response:")
        st.write(final_response)

        if image_file:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)

        st.markdown("📝 **Was this response helpful?**")
        st.radio("Feedback:", ["👍 Yes", "👎 No", "😐 Not sure"])

if __name__ == "__main__":
    main()
