"""
CUSTOMER REVIEW SENTIMENT ANALYSIS - training script
----------------------------------------------------
Dataset : Amazon Fine Food Reviews (Reviews.csv)
Columns : Id, ProductId, UserId, ProfileName, HelpfulnessNumerator,
          HelpfulnessDenominator, Score, Time, Summary, Text

Run     : python code.py
Output  : model/sentiment_model.joblib   (used by app.py)
          outputs/*.png, outputs/*.csv   (charts and metrics for your report)
"""

import os
import re

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files, no pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ----------------------------- settings ------------------------------------
DATA_PATH = "Reviews.csv"
SAMPLE_SIZE = 100_000   # set to None to use all ~568K rows (slower)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_DIR = "model"
OUTPUT_DIR = "outputs"
LABELS = ["Negative", "Neutral", "Positive"]

# Keep negation words: "not good" must stay different from "good"
NEGATIONS = {"not", "no", "nor", "never", "cannot", "nothing", "none"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - NEGATIONS


# ----------------------------- text cleaning -------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip HTML/URLs/punctuation, drop stopwords."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)            # html tags like <br />
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # urls
    text = re.sub(r"n't\b", " not", text)           # don't -> do not
    text = re.sub(r"[^a-z\s]", " ", text)           # keep letters only
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)


def score_to_label(score: int) -> str:
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


# ----------------------------- data ----------------------------------------
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"'{DATA_PATH}' not found. Put Reviews.csv in the same folder as code.py."
        )

    # Only read the columns we need (also ignores the first index column)
    wanted = {"Score", "Summary", "Text"}
    df = pd.read_csv(DATA_PATH, usecols=lambda c: c in wanted)

    missing = {"Score", "Text"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    if "Summary" not in df.columns:
        df["Summary"] = ""

    df = df.dropna(subset=["Score", "Text"])
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score"])
    df["Score"] = df["Score"].astype(int)
    df = df[df["Score"].between(1, 5)]
    df["Summary"] = df["Summary"].fillna("")

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    df["label"] = df["Score"].apply(score_to_label)
    # Summary is short and very informative, so combine it with the full text
    df["raw_text"] = df["Summary"].astype(str) + " " + df["Text"].astype(str)
    return df.reset_index(drop=True)


# ----------------------------- plots ---------------------------------------
def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df, order=LABELS, palette=["#e15759", "#bab0ab", "#59a14f"])
    plt.title("Sentiment class distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def save_top_words(vectorizer, lr_model, n: int = 20) -> None:
    """Most positive / negative words according to Logistic Regression."""
    names = vectorizer.get_feature_names_out()
    rows = []
    for i, cls in enumerate(lr_model.classes_):
        top = lr_model.coef_[i].argsort()[-n:][::-1]
        for rank, idx in enumerate(top, 1):
            rows.append({"class": cls, "rank": rank, "word": names[idx]})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "top_words.csv"), index=False)


# ----------------------------- main ----------------------------------------
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1/5 Loading data ...")
    df = load_data()
    print(f"    Rows used: {len(df):,}")
    print(df["label"].value_counts().to_string())
    plot_class_distribution(df)

    print("2/5 Cleaning text (this can take a minute) ...")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    print("3/5 Splitting and vectorizing (TF-IDF) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3)
    X_train_t = vectorizer.fit_transform(X_train)
    X_test_t = vectorizer.transform(X_test)

    print("4/5 Training models ...")
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=2.0
        ),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
    }

    results, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train_t, y_train)
        pred = model.predict(X_test_t)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        results.append({"model": name, "accuracy": round(acc, 4), "macro_f1": round(f1, 4)})
        fitted[name] = (model, pred)
        print(f"    {name:20s} accuracy={acc:.4f}  macro-F1={f1:.4f}")

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    best_name = results_df.iloc[0]["model"]
    best_model, best_pred = fitted[best_name]
    print(f"\nBest model: {best_name}\n")
    report = classification_report(y_test, best_pred, labels=LABELS, digits=3)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n\n{report}")

    plot_confusion(y_test, best_pred, best_name)
    save_top_words(vectorizer, fitted["Logistic Regression"][0])

    print("5/5 Saving model ...")
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": best_model,
            "model_name": best_name,
            "accuracy": float(results_df.iloc[0]["accuracy"]),
            "macro_f1": float(results_df.iloc[0]["macro_f1"]),
        },
        os.path.join(MODEL_DIR, "sentiment_model.joblib"),
    )
    print(f"Done. Saved to {MODEL_DIR}/sentiment_model.joblib")
    print(f"Charts and metrics are in the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    main()
