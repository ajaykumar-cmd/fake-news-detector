import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

import os
import joblib
import re
import string
import nltk
import traceback
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Check files exist
if not os.path.exists("model.pkl"):
    st.error("❌ model.pkl not found")
    st.stop()

if not os.path.exists("tfidf_vectorizer.pkl"):
    st.error("❌ tfidf_vectorizer.pkl not found")
    st.stop()

# Load model and vectorizer
try:
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
except Exception:
    st.error(traceback.format_exc())
    st.stop()

# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^\\x00-\\x7F]+", " ", text)
    text = re.sub(r"\\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def preprocess(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# UI
st.title("📰 Fake News Detection App")
st.write("Paste a news article below to check whether it's Fake or Real.")

user_input = st.text_area("Enter news text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vector = tfidf.transform([processed])

        prediction = model.predict(vector)[0]

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(vector)[0]) * 100
            st.write(f"**Confidence:** {confidence:.2f}%")

        if str(prediction).upper() == "FAKE":
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")