import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

import os
import re
import string
import traceback
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Debug information
st.write("Current directory:", os.getcwd())
st.write("Files:", os.listdir("."))

# Load model
try:
    model = joblib.load("model.pkl")
    st.success("✅ model.pkl loaded")
except Exception:
    st.code(traceback.format_exc())
    st.stop()

# Load vectorizer
try:
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    st.success("✅ tfidf_vectorizer.pkl loaded")
except Exception:
    st.code(traceback.format_exc())
    st.stop()

# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter news text")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vector = tfidf.transform([processed])
        prediction = model.predict(vector)[0]

        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(vector)[0]) * 100
            st.write(f"Confidence: {confidence:.2f}%")

        if str(prediction).upper() == "FAKE":
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")