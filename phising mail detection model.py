
"""
Phishing Email Detection Model (Mini Project)
==============================================
Trains a Scikit-learn model to classify emails as "Phishing" or "Safe"
based on their text content and URL/keyword features, then reports
accuracy and a confusion matrix.

Pipeline:
  1. Load a labeled phishing/legitimate email dataset
     - Tries to download a public dataset (Hugging Face mirror of the
       Kaggle "Phishing Email Detection" dataset) automatically.
     - Falls back to a locally generated synthetic dataset if the
       download isn't available (e.g. no internet access), so the
       script always runs end-to-end.
  2. Extract features:
     - TF-IDF vector of the email text
     - Extra numeric features: URL count, exclamation marks, urgency
       keyword count, capital-letter ratio, text length
  3. Train a classifier (Logistic Regression) on a train/test split
  4. Evaluate: accuracy, precision/recall/F1, and a confusion matrix
     (printed to console + saved as a PNG image)
  5. Let you classify your own custom email text from the command line

Usage:
    python phishing_email_detector.py
    python phishing_email_detector.py --max-samples 5000
    python phishing_email_detector.py --classify "Dear user, verify your account now: http://bit.ly/xyz"

Requirements (install once):
    pip install pandas numpy scikit-learn matplotlib requests
"""

import argparse
import io
import re
import sys
import random

import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")  # safe for headless/script use
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

DATASET_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset"
    "/resolve/main/Phishing_Email.csv"
)

# Possible column name variants across dataset mirrors
TEXT_COLUMNS = ["Email Text", "text_combined", "text", "body", "Body", "Text"]
LABEL_COLUMNS = ["Email Type", "label", "Label", "Category", "class"]

PHISHING_LABELS = {"phishing email", "phishing", "1", "spam", 1, True}
SAFE_LABELS = {"safe email", "legitimate", "0", "ham", 0, False}


def try_download_dataset(max_samples=None):
    """Attempt to download the public phishing-email dataset. Returns a
    DataFrame with columns ['text', 'label'] (label: 1=phishing, 0=safe),
    or None if the download/parse fails."""
    try:
        import requests
        print(f"[*] Attempting to download dataset from:\n    {DATASET_URL}")
        resp = requests.get(DATASET_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as e:
        print(f"[!] Dataset download failed ({e}). Will use a synthetic dataset instead.")
        return None

    text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)

    if text_col is None or label_col is None:
        print(f"[!] Downloaded dataset has unexpected columns {list(df.columns)}. "
              f"Falling back to synthetic dataset.")
        return None

    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]

    def normalize_label(v):
        key = v.strip().lower() if isinstance(v, str) else v
        if key in PHISHING_LABELS:
            return 1
        if key in SAFE_LABELS:
            return 0
        return None

    df["label"] = df["label"].apply(normalize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)

    if len(df) < 20:
        print("[!] Downloaded dataset had too few usable rows. Falling back to synthetic dataset.")
        return None

    if max_samples and len(df) > max_samples:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_samples // 2), random_state=42)
        )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    print(f"[*] Loaded {len(df)} emails from the public dataset "
          f"({(df['label'] == 1).sum()} phishing / {(df['label'] == 0).sum()} safe).")
    return df


def generate_synthetic_dataset(n_per_class=400, seed=42):
    """Generate a synthetic-but-realistic phishing/legitimate email dataset,
    used as a fallback when a live dataset download isn't available."""
    rnd = random.Random(seed)

    phishing_openers = [
        "Dear Valued Customer,", "Dear User,", "Attention Account Holder,",
        "Hello,", "Urgent Notice:",
    ]
    phishing_bodies = [
        "We have detected unusual activity on your account. Verify your identity immediately at {url} or your account will be suspended within 24 hours.",
        "Your payment could not be processed. Update your billing information now at {url} to avoid service interruption.",
        "Congratulations! You have won a prize. Claim your reward within 48 hours by clicking {url} and entering your bank details.",
        "Your password will expire today. Click {url} now to reset it and keep access to your account.",
        "Security Alert: Someone tried to log into your account from a new device. Confirm it was you at {url} immediately.",
        "Your package could not be delivered. Confirm your address and pay a small fee at {url} to reschedule delivery.",
        "IRS Notice: You have an unpaid tax balance. Settle it immediately at {url} to avoid legal action.",
        "Your subscription has been cancelled due to a billing issue. Reactivate now at {url} before it's too late.",
    ]
    phishing_urls = [
        "http://bit.ly/2xk9Az", "http://verify-account-secure.com/login",
        "http://192.168.4.22/reset", "http://paypal-security-check.net",
        "http://amaz0n-support.info/verify", "http://bankalert-secure.ru/login",
    ]
    phishing_closers = [
        "Failure to act immediately will result in permanent suspension.",
        "This is your final notice.",
        "Act now, offer expires soon!",
        "Do not ignore this urgent message.",
    ]

    legit_openers = [
        "Hi Team,", "Hello,", "Good morning,", "Dear Colleague,", "Hi there,",
    ]
    legit_bodies = [
        "Attached is the report you requested for this quarter's sales figures. Let me know if you have any questions.",
        "Just a reminder that our meeting is scheduled for Thursday at 10am in the main conference room.",
        "Thanks for your email. I'll review the document and get back to you by end of week.",
        "The project timeline has been updated. Please check the shared drive for the latest version.",
        "Here are the notes from today's stand-up. Let me know if I missed anything.",
        "Your order has shipped and should arrive within 3-5 business days. You can track it from your account dashboard.",
        "Reminder: your subscription renews next month. No action is needed unless you want to make changes.",
        "Please find attached the invoice for last month's services. Let us know if you have any billing questions.",
    ]
    legit_closers = [
        "Best regards,", "Thanks,", "Talk soon,", "Kind regards,",
    ]

    rows = []
    for _ in range(n_per_class):
        url = rnd.choice(phishing_urls)
        text = (
            f"{rnd.choice(phishing_openers)} "
            f"{rnd.choice(phishing_bodies).format(url=url)} "
            f"{rnd.choice(phishing_closers)}"
        )
        rows.append({"text": text, "label": 1})

    for _ in range(n_per_class):
        text = (
            f"{rnd.choice(legit_openers)} "
            f"{rnd.choice(legit_bodies)} "
            f"{rnd.choice(legit_closers)} Alex"
        )
        rows.append({"text": text, "label": 0})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[*] Generated a synthetic dataset with {len(df)} emails "
          f"({n_per_class} phishing / {n_per_class} safe).")
    return df


def load_dataset(max_samples=None, force_synthetic=False):
    if not force_synthetic:
        df = try_download_dataset(max_samples=max_samples)
        if df is not None:
            return df
    return generate_synthetic_dataset(n_per_class=(max_samples // 2) if max_samples else 400)


# ---------------------------------------------------------------------------
# 2. Feature extraction
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
URGENCY_KEYWORDS = [
    "urgent", "immediately", "verify", "suspend", "suspended", "confirm",
    "click", "act now", "password", "account", "limited time", "winner",
    "congratulations", "bank", "security alert", "update your",
]


def extract_numeric_features(texts):
    """Build extra numeric features per email: URL count, exclamation
    marks, urgency-keyword hits, capital-letter ratio, and text length."""
    feats = []
    for t in texts:
        t_lower = t.lower()
        url_count = len(URL_PATTERN.findall(t))
        exclam_count = t.count("!")
        urgency_hits = sum(t_lower.count(k) for k in URGENCY_KEYWORDS)
        letters = [c for c in t if c.isalpha()]
        caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        length = len(t)
        feats.append([url_count, exclam_count, urgency_hits, caps_ratio, length])
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# 3. Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df, output_prefix="phishing_model"):
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"].tolist(), df["label"].tolist(),
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("[*] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    print("[*] Extracting numeric features (URLs, urgency keywords, etc.)...")
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(extract_numeric_features(X_train_text))
    X_test_numeric = scaler.transform(extract_numeric_features(X_test_text))

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_numeric)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_numeric)])

    print("[*] Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"])

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("                 Pred: Safe   Pred: Phishing")
    print(f"Actual: Safe       {cm[0][0]:<10} {cm[0][1]}")
    print(f"Actual: Phishing   {cm[1][0]:<10} {cm[1][1]}")
    print("\nClassification Report:")
    print(report)

    # Save confusion matrix as an image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (Accuracy: {acc * 100:.1f}%)")
    plt.tight_layout()
    cm_path = f"{output_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n[*] Confusion matrix image saved to {cm_path}")

    return model, vectorizer, scaler


def classify_email(model, vectorizer, scaler, text):
    tfidf_feat = vectorizer.transform([text])
    numeric_feat = scaler.transform(extract_numeric_features([text]))
    combined = hstack([tfidf_feat, csr_matrix(numeric_feat)])
    pred = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0]
    label = "Phishing" if pred == 1 else "Safe"
    confidence = proba[pred] * 100
    return label, confidence


# ---------------------------------------------------------------------------
# 4. CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phishing Email Detection Model (Scikit-learn)")
    parser.add_argument("--max-samples", type=int, default=4000,
                         help="Cap the number of emails used for training (keeps runtime reasonable)")
    parser.add_argument("--synthetic", action="store_true",
                         help="Skip the download and use the synthetic dataset directly")
    parser.add_argument("--classify", type=str, default=None,
                         help="After training, classify a custom piece of email text")
    args = parser.parse_args()

    df = load_dataset(max_samples=args.max_samples, force_synthetic=args.synthetic)
    model, vectorizer, scaler = train_and_evaluate(df)

    if args.classify:
        label, confidence = classify_email(model, vectorizer, scaler, args.classify)
        print("\n" + "-" * 60)
        print("CUSTOM EMAIL CLASSIFICATION")
        print("-" * 60)
        print(f"Text: {args.classify[:200]}")
        print(f"Prediction: {label} ({confidence:.1f}% confidence)")
    else:
        # Demonstrate with a couple of built-in examples
        examples = [
            "Dear user, your account has been suspended. Click here immediately to verify: http://secure-verify-now.com",
            "Hi Sarah, attached is the agenda for tomorrow's meeting. Let me know if you'd like to add anything. Thanks, John",
        ]
        print("\n" + "-" * 60)
        print("SAMPLE CLASSIFICATIONS")
        print("-" * 60)
        for ex in examples:
            label, confidence = classify_email(model, vectorizer, scaler, ex)
            print(f"[{label} ({confidence:.1f}%)] {ex[:80]}...")


if __name__ == "__main__":
    main()