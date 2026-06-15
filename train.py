"""
train.py  -  Downloads a real fake-news dataset and trains a high-accuracy detector.

Tries (in order):
 1. liar  dataset via HuggingFace (new parquet-based version)
 2. GossipCop/PolitiFact via FakeNewsNet on HuggingFace
 3. Direct CSV download from a reliable public source
 4. Local fake_news_dataset.csv (tiny fallback)

Pipeline:
  - Text Preprocessor
  - TF-IDF (word n-grams 1-3) + TF-IDF (char n-grams 3-6)  [FeatureUnion]
  - LinearSVC + CalibratedClassifierCV

Run:
    python -X utf8 train.py
"""

import os
import sys
import io
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from utils.preprocessor import clean_text
from utils.fact_builder import generate_general_knowledge

# ─────────────────────────────────────────────────────────────────────────────
#   1.  Data Loaders
# ─────────────────────────────────────────────────────────────────────────────

LIAR_LABEL_MAP = {
    # label int -> binary (0=FAKE, 1=REAL)
    # The parquet version uses strings
    "false":       0,
    "half-true":   1,
    "mostly-true": 1,
    "true":        1,
    "barely-true": 0,
    "pants-fire":  0,
    # numeric fallback (older API)
    0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0,
}


def try_load_liar_huggingface():
    """Try the new HuggingFace parquet-based LIAR dataset."""
    print("Trying HuggingFace LIAR dataset (parquet) ...")
    from datasets import load_dataset
    ds = load_dataset("liar", trust_remote_code=False)
    rows = []
    for split in ds.keys():
        for ex in ds[split]:
            raw_label = ex["label"]
            label_bin = LIAR_LABEL_MAP.get(raw_label, -1)
            if label_bin == -1:
                continue
            combined = " ".join(filter(None, [
                str(ex.get("statement", "")),
                str(ex.get("subject",   "")),
                str(ex.get("speaker",   "")),
                str(ex.get("context",   "")),
            ]))
            rows.append({"text": combined, "label": label_bin})
    df = pd.DataFrame(rows)
    print(f"   LIAR loaded: {len(df)} samples")
    return df


def try_load_csv_from_url():
    """
    Download the WELFake dataset (72,000 article headlines + text, CC licensed).
    Hosted on multiple public mirrors.
    """
    urls = [
        # WELFake CSV - label: 0=Fake, 1=Real
        "https://raw.githubusercontent.com/sumeetkr/AwesomeFakeNews/main/WELFake_Dataset.csv",
        # Backup: Kaggle-style minimal fake news CSV hosted on GitHub
        "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv",
    ]
    for url in urls:
        try:
            print(f"Downloading dataset from:\n   {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"   Downloaded: {len(df)} rows, columns: {list(df.columns)}")

            # Normalize columns
            df.columns = [c.lower().strip() for c in df.columns]

            # Unified label parsing for any CSV that has a 'label' column
            if "label" in df.columns:
                # Find best text column
                text_col = None
                for col in ["text", "title", "body", "content"]:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col is None:
                    continue

                # Map string labels (FAKE/REAL) OR keep numeric (0/1)
                raw_labels = df["label"].astype(str).str.strip().str.upper()
                if raw_labels.isin(["FAKE", "REAL"]).any():
                    df["label"] = raw_labels.map({"FAKE": 0, "REAL": 1})
                else:
                    df["label"] = pd.to_numeric(df["label"], errors="coerce")

                df = df[[text_col, "label"]].rename(columns={text_col: "text"})
                df.dropna(subset=["label"], inplace=True)
                df["label"] = df["label"].astype(int)

                if df["label"].nunique() == 2 and len(df) > 50:
                    print(f"   Loaded {len(df)} cleaned samples.")
                    return df


        except Exception as e:
            print(f"   Failed: {e}")

    return None


def load_local_csv():
    print("Using local fake_news_dataset.csv (accuracy will be limited).")
    df = pd.read_csv("fake_news_dataset.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    return df[["text", "label"]]


# ─────────────────────────────────────────────────────────────────────────────
#   2.  Model Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline():
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        stop_words="english",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    features = FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec),
    ])
    svc = LinearSVC(C=0.8, class_weight="balanced", max_iter=2000, dual=True)
    calibrated = CalibratedClassifierCV(svc, cv=3, method="sigmoid")

    return Pipeline([("features", features), ("clf", calibrated)])


# ─────────────────────────────────────────────────────────────────────────────
#   3.  Train
# ─────────────────────────────────────────────────────────────────────────────

def train():
    df = None

    # --- Try each data source in order ---
    for loader in [try_load_liar_huggingface, try_load_csv_from_url]:
        try:
            df = loader()
            if df is not None and len(df) > 50:
                break
        except Exception as e:
            print(f"   Loader failed: {e}")

    if df is None or len(df) <= 50:
        df = load_local_csv()

    # --- Augment with general knowledge facts ---
    print("Augmenting with general knowledge facts (Animals, Capitals, Physics, etc)...")
    facts_df = generate_general_knowledge()
    df = pd.concat([df, facts_df], ignore_index=True)

    # --- Clean ---
    print("Cleaning text ...")
    df = df.copy()
    df["text"] = df["text"].fillna("").apply(clean_text)
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    print(f"Dataset: {len(df)} samples  "
          f"| FAKE={int((df.label==0).sum())}  "
          f"| REAL={int((df.label==1).sum())}")

    # --- Balance ---
    fake = df[df.label == 0]
    real = df[df.label == 1]
    n    = min(len(fake), len(real))
    fake = resample(fake, n_samples=n, random_state=42)
    real = resample(real, n_samples=n, random_state=42)
    df   = pd.concat([fake, real]).sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, df["label"].values,
        test_size=0.15, random_state=42, stratify=df["label"].values
    )
    print(f"Training on {len(X_train)} | Testing on {len(X_test)} ...")

    # --- Train ---
    clf = build_pipeline()
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    preds = clf.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy : {acc*100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["FAKE", "REAL"]))

    # --- Save ---
    os.makedirs("models", exist_ok=True)
    pickle.dump(clf, open("models/pipeline.pkl", "wb"))
    print("Saved: models/pipeline.pkl")
    return acc


if __name__ == "__main__":
    train()
