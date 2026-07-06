import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------- Download NLTK resources (safe to call every run, cached after first) ----------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ---------- Load Model & Vectorizer ----------
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------- Text Cleaning (identical to training pipeline) ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Paste a news article below to check whether it's likely **Fake** or **Real**.")

user_input = st.text_area("Enter news text:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text before predicting.")
    else:
        final_text = preprocess(user_input)
        vectorized = tfidf.transform([final_text])
        prediction = model.predict(vectorized)[0]  # returns 'FAKE' or 'REAL' directly

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vectorized)[0]
            confidence = max(proba) * 100
        else:
            confidence = None

        if str(prediction).upper() == "FAKE":
            st.error("🚨 This looks like **FAKE** news.")
        else:
            st.success("✅ This looks like **REAL** news.")

        if confidence is not None:
            st.write(f"Confidence: **{confidence:.2f}%**")
