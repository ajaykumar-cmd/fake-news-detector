import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Cache model loading
@st.cache_resource
def load_files():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, tfidf = load_files()

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Text cleaning
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

# Preprocessing
def preprocess(text):
    cleaned = clean_text(text)

    # Simple tokenization (avoids punkt/punkt_tab dependency)
    tokens = cleaned.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# UI
st.title("📰 Fake News Detection App")
st.write("Paste a news article below to check whether it is Fake or Real.")

user_input = st.text_area("Enter news text", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        processed = preprocess(user_input)
        vector = tfidf.transform([processed])
        prediction = model.predict(vector)[0]

        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vector).max() * 100
            st.write(f"### Confidence: {confidence:.2f}%")

        if str(prediction).strip().upper() == "FAKE":
            st.error("🚨 Prediction: Fake News")
        else:
            st.success("✅ Prediction: Real News")