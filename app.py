"""
CUSTOMER REVIEW SENTIMENT ANALYSIS - Streamlit app
Run: streamlit run app.py   (after running `python code.py` once to train the model)
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_PATH = os.path.join("model", "sentiment_model.joblib")

# Same cleaning as code.py (kept here so app.py does not import code.py)
NEGATIONS = {"not", "no", "nor", "never", "cannot", "nothing", "none"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS
EMOJI = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict_proba(bundle, texts):
    """Return (classes, probability matrix) for any of the saved models."""
    cleaned = [clean_text(t) for t in texts]
    X = bundle["vectorizer"].transform(cleaned)
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    else:  # LinearSVC: turn decision scores into pseudo-probabilities (softmax)
        scores = model.decision_function(X)
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
    return list(model.classes_), probs


# ------------------------------- UI ----------------------------------------
st.set_page_config(page_title="Customer Review Sentiment Analysis", page_icon="💬", layout="centered")
st.title("💬 Customer Review Sentiment Analysis")
st.caption("Classifies customer reviews as Positive, Neutral or Negative.")

bundle = load_bundle()
if bundle is None:
    st.error("Model not found. Run `python code.py` first to train and save the model.")
    st.stop()

with st.sidebar:
    st.header("Model info")
    st.write(f"**Model:** {bundle['model_name']}")
    st.write(f"**Accuracy:** {bundle['accuracy']:.2%}")
    st.write(f"**Macro F1:** {bundle['macro_f1']:.2%}")
    st.caption("Trained on Amazon Fine Food Reviews.")

tab_single, tab_batch = st.tabs(["Single review", "Batch (CSV upload)"])

# ---- single review ----
with tab_single:
    examples = {
        "(write your own)": "",
        "Positive example": "Absolutely love this coffee! Rich flavor and arrived fast. Will buy again.",
        "Negative example": "Terrible taste and the package was damaged. Not worth the money.",
        "Neutral example": "It is okay. Nothing special, but not bad either.",
    }
    choice = st.selectbox("Try an example", list(examples.keys()))
    review = st.text_area("Enter a customer review", value=examples[choice], height=150)

    if st.button("Analyze sentiment", type="primary"):
        if not review.strip():
            st.warning("Please enter a review first.")
        else:
            classes, probs = predict_proba(bundle, [review])
            p = probs[0]
            label = classes[int(p.argmax())]
            st.subheader(f"{EMOJI.get(label, '')} {label}")
            st.write(f"Confidence: **{p.max():.1%}**")
            st.bar_chart(pd.DataFrame({"probability": p}, index=classes))

# ---- batch ----
with tab_batch:
    st.write("Upload a CSV with a column of review text (e.g. the `Text` column of Reviews.csv).")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        default = list(data.columns).index("Text") if "Text" in data.columns else 0
        col = st.selectbox("Review text column", data.columns, index=default)
        max_rows = st.slider("Rows to analyze", 10, min(len(data), 20000), min(len(data), 1000))

        if st.button("Analyze file"):
            subset = data.head(max_rows).copy()
            classes, probs = predict_proba(bundle, subset[col].fillna("").astype(str).tolist())
            subset["predicted_sentiment"] = [classes[i] for i in probs.argmax(axis=1)]
            subset["confidence"] = probs.max(axis=1).round(3)

            counts = subset["predicted_sentiment"].value_counts()
            st.subheader("Sentiment summary")
            st.bar_chart(counts)
            st.dataframe(subset.head(100))
            st.download_button(
                "Download results (CSV)",
                subset.to_csv(index=False).encode("utf-8"),
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
