import streamlit as st
import os
import base64
from PIL import Image
import requests
from io import BytesIO


GOOGLE_API_KEY = "AIzaSyA0JpFuXyv64fViTZIXRVsnqOoyuzPUKBs"

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
    res = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        st.error(f"Gemini API Error: {res.status_code} - {res.text}")
        return "❌ Error processing the input."

def generate_image_from_text(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    res = requests.post(IMAGE_GEN_URL, headers=headers, json=payload)
    if res.status_code == 200:
        image_data = res.json()['candidates'][0]['content']['parts'][0]['inlineData']['data']
        return image_data
    else:
        st.error(f"Image generation failed: {res.status_code} - {res.text}")
        return None

def main():
    st.set_page_config(page_title="🩺 Multimodal Medical Chatbot", layout="wide")
    st.title("🧠 Gemini-Powered Medical Chatbot")
    st.caption("Understand and generate text + images for medical questions.")

    with st.form("med_form"):
        user_query = st.text_area("🔎 Ask a medical question or describe symptoms:")
        image_file = st.file_uploader("🖼 Upload a medical image (optional)", type=["jpg", "jpeg", "png"])
        generate_visual = st.checkbox("🖌 Generate an illustration based on your question?")
        submitted = st.form_submit_button("Submit")

    if submitted and user_query:
        img_data = encode_image(image_file) if image_file else None
        response = query_gemini(user_query, img_data)

        st.subheader("💬 Response from Gemini")
        st.write(response)

        if generate_visual:
            st.subheader("🎨 Generated Medical Illustration")
            image_base64 = generate_image_from_text(user_query)
            if image_base64:
                st.image(BytesIO(base64.b64decode(image_base64)), use_column_width=True)

if __name__ == "__main__":
    main()
