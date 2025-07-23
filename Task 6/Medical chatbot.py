import streamlit as st
import os
import base64
from PIL import Image
import requests
from io import BytesIO
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("AIzaSyA0JpFuXyv64fViTZIXRVsnqOoyuzPUKBs")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent"
IMAGE_GEN_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagegeneration:generateContent"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": GOOGLE_API_KEY
}

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

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

def generate_image_from_text(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        res = requests.post(IMAGE_GEN_URL, headers=headers, json=payload)
        res.raise_for_status()
        image_data = res.json()['candidates'][0]['content']['parts'][0]['inlineData']['data']
        return image_data
    except Exception as e:
        st.error(f"Image generation failed: {res.status_code} - {res.text}")
        return None

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_empathy_prefix(sentiment):
    if sentiment == "positive":
        return "😊 I'm glad to hear that! Here's some helpful information:\n\n"
    elif sentiment == "negative":
        return "😟 I'm sorry you're experiencing this. Let me help as best I can:\n\n"
    else:
        return "🙂 Here's what I found based on your input:\n\n"

def main():
    st.set_page_config(page_title="🩺 Multimodal Medical Chatbot", layout="wide")
    st.title("🧠 Gemini-Powered Medical Chatbot with Sentiment Awareness")
    st.caption("Understand and respond to medical questions with emotional intelligence.")

    with st.form("med_form"):
        user_query = st.text_area("🔎 Ask a medical question or describe symptoms:")
        image_file = st.file_uploader("🖼 Upload a medical image (optional)", type=["jpg", "jpeg", "png"])
        generate_visual = st.checkbox("🖌 Generate an illustration based on your question?")
        submitted = st.form_submit_button("Submit")

    if submitted and user_query:
        sentiment = analyze_sentiment(user_query)
        prefix = generate_empathy_prefix(sentiment)
        img_data = encode_image(image_file) if image_file else None

        with st.spinner("Processing your request..."):
            gemini_response = query_gemini(user_query, img_data)
            final_response = prefix + gemini_response

        emoji_map = {
            "positive": "😊",
            "neutral": "😐",
            "negative": "😢"
        }

        st.subheader(f"💬 Response from Gemini (Sentiment: {emoji_map[sentiment]} `{sentiment}`)")
        st.write(final_response)

        if image_file:
            st.image(image_file, caption="📷 Uploaded Image", use_column_width=True)

        if generate_visual:
            st.subheader("🎨 Generated Medical Illustration")
            image_base64 = generate_image_from_text(user_query)
            if image_base64:
                st.image(BytesIO(base64.b64decode(image_base64)), use_column_width=True)

        st.markdown("📝 **Was this response helpful?**")
        feedback = st.radio("Select an option:", ["👍 Yes", "👎 No", "😐 Not sure"])

if __name__ == "__main__":
    main()
