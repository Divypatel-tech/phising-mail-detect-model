[README_phishing_model.md](https://github.com/user-attachments/files/32695175/README_phishing_model.md)
# Phishing Email Detection Model

A machine learning model built with **Scikit-learn** that classifies emails as **Phishing** or **Safe**, based on their text content, embedded URLs, and linguistic patterns (urgency keywords, punctuation, capitalization). Built as a mini project to demonstrate a full text-classification pipeline: data loading, feature engineering, training, and evaluation.

## Features

- **Automatic dataset loading** — downloads a public phishing/legitimate email dataset (18.7k labeled emails); falls back to a locally generated synthetic dataset if no internet connection is available, so the script always runs end-to-end.
- **Text feature extraction** — TF-IDF vectorization of email content (unigrams + bigrams).
- **Handcrafted feature extraction** — URL count, exclamation marks, urgency-keyword hits, capital-letter ratio, and text length.
- **Model training** — Logistic Regression classifier with class balancing.
- **Evaluation** — accuracy, precision/recall/F1 per class, and a confusion matrix (printed + saved as an image).
- **Model persistence** — the trained model, vectorizer, and scaler are saved to disk so they can be reloaded without retraining.
- **Custom classification** — classify any email text you provide from the command line.

## Requirements

```bash
pip install pandas numpy scikit-learn matplotlib requests joblib
```

(`scipy` is installed automatically as a Scikit-learn dependency.)

## Usage

```bash
python phishing_email_detector.py
```

### Optional arguments

| Argument         | Description                                                        | Default |
|------------------|---------------------------------------------------------------------|---------|
| `--max-samples`  | Cap the number of emails used for training (keeps runtime reasonable) | `4000`  |
| `--synthetic`    | Skip the download and use the synthetic dataset directly              | Off     |
| `--classify`     | After training, classify a custom piece of email text                 | None    |

### Examples

```bash
# Train on the downloaded dataset (default)
python phishing_email_detector.py

# Limit dataset size for a faster run
python phishing_email_detector.py --max-samples 2000

# Skip the download and use the synthetic dataset
python phishing_email_detector.py --synthetic

# Classify your own email text after training
python phishing_email_detector.py --classify "Your account will be suspended, click here immediately to verify"
```

## Outputs

Running the script produces:

- **Console output** — dataset summary, accuracy, confusion matrix (as text), and classification report.
- **`phishing_model_confusion_matrix.png`** — a heatmap image of the confusion matrix.
- **`phishing_model.pkl`** — the trained model, TF-IDF vectorizer, and feature scaler, saved together so they can be reloaded later without retraining.

## How It Works (Full Explanation)

### 1. Getting the Data

```python
DATASET_URL = "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv"
```

The script downloads a public, no-login-required mirror of a Kaggle phishing-email dataset (~18.7k emails) directly into memory:

```python
resp = requests.get(DATASET_URL, timeout=30)
df = pd.read_csv(io.BytesIO(resp.content))
```

Nothing is saved to disk during this step — the CSV bytes are held in memory just long enough for pandas to parse them into a DataFrame.

Since different dataset mirrors sometimes use different column names (e.g. `"Email Text"` vs `"text_combined"`, `"Email Type"` vs `"label"`), the script checks a list of known possibilities and normalizes whichever it finds into two consistent columns: `text` and `label` (`1` = phishing, `0` = safe).

**Fallback:** if the download fails for any reason (no internet, firewall, URL changes), `generate_synthetic_dataset()` builds a template-based dataset instead — phishing-style emails (urgent language, fake links) and legitimate-style emails (normal workplace correspondence) — so the script never crashes for lack of connectivity.

### 2. Turning Text Into Features a Model Can Use

Machine learning models need numbers, not sentences. Two feature sets are extracted and combined side-by-side for every email:

**a) TF-IDF (Term Frequency–Inverse Document Frequency)**

```python
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
```

Each email becomes a vector of word/phrase importance scores. A word that appears often in one email but rarely across the whole dataset scores higher (it's distinctive to that email); very common words ("the", "and") are filtered out via `stop_words="english"`. `ngram_range=(1, 2)` captures both single words and two-word phrases (e.g. "click here", "verify account").

**b) Handcrafted numeric features**

```python
def extract_numeric_features(texts):
    url_count = ...      # number of http(s)/www links in the text
    exclam_count = ...   # number of "!" characters
    urgency_hits = ...   # count of urgency keywords ("verify", "suspended", "immediately", etc.)
    caps_ratio = ...     # fraction of letters that are uppercase
    length = ...          # total character length
```

These directly encode the patterns the task description asks for — URL presence and keyword-based signals. Phishing emails tend to have more embedded links, more urgency language, and occasionally more aggressive capitalization.

These numeric features are scaled with `StandardScaler` (so "length in characters," which can be in the hundreds, doesn't dominate "URL count," which is typically 0–5) and then joined to the TF-IDF matrix using `scipy.sparse.hstack`, producing one combined feature matrix per email.

### 3. Training the Model

```python
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
```

- **80/20 split** — 80% of emails train the model; the remaining 20% are held back purely for testing, so evaluation reflects performance on unseen data.
- **`stratify=df["label"]`** — keeps the phishing/safe ratio consistent across both the train and test sets.
- **Logistic Regression** — a simple, fast, well-understood classifier that works well on high-dimensional sparse text features like TF-IDF. It learns which words and features push a prediction toward "phishing" vs. "safe."
- **`class_weight="balanced"`** — automatically adjusts for any imbalance between phishing and safe emails, so the model doesn't just learn to predict the majority class.

### 4. Evaluating the Model

```python
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
```

- **Accuracy** — the percentage of test emails classified correctly.
- **Confusion matrix** — a 2×2 grid showing exactly how predictions broke down:

|                      | Predicted: Safe             | Predicted: Phishing          |
|----------------------|------------------------------|-------------------------------|
| **Actual: Safe**     | Correctly identified as safe | False positive (safe email flagged as phishing) |
| **Actual: Phishing** | False negative (phishing email missed) | Correctly identified as phishing |

The diagonal (top-left → bottom-right) represents correct predictions; the off-diagonal represents mistakes. A false negative (a phishing email getting through) is generally the more dangerous type of error in this application.

- **Classification report** — adds precision, recall, and F1-score per class, since overall accuracy alone can hide whether the model is better at catching phishing emails or safe ones.

The confusion matrix is also rendered visually and saved as an image:

```python
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.savefig("phishing_model_confusion_matrix.png", dpi=150)
```

### 5. Saving and Reusing the Model

```python
joblib.dump({"model": model, "vectorizer": vectorizer, "scaler": scaler}, "phishing_model.pkl")
```

The trained model, TF-IDF vectorizer, and feature scaler are bundled together and saved to disk, so the model can be reloaded later and used to classify new emails without retraining from scratch.

### 6. Classifying New Emails

```python
def classify_email(model, vectorizer, scaler, text):
    tfidf_feat = vectorizer.transform([text])
    numeric_feat = scaler.transform(extract_numeric_features([text]))
    combined = hstack([tfidf_feat, csr_matrix(numeric_feat)])
    pred = model.predict(combined)[0]
```

Any new email text goes through the exact same feature pipeline used during training — note `.transform()` rather than `.fit_transform()`, since the vectorizer and scaler must stay fixed to whatever they learned during training, not be refit on new data. The model then predicts a label, and `predict_proba()` provides a confidence percentage alongside it.

## Sample Output

```
[*] Loaded 4000 emails from the public dataset (2013 phishing / 1987 safe).
[*] Extracting TF-IDF features...
[*] Extracting numeric features (URLs, urgency keywords, etc.)...
[*] Training Logistic Regression classifier...

============================================================
MODEL EVALUATION
============================================================
Accuracy: 96.75%

Confusion Matrix (rows=actual, cols=predicted):
                 Pred: Safe   Pred: Phishing
Actual: Safe       385        13
Actual: Phishing   13         389

Classification Report:
              precision    recall  f1-score   support

        Safe       0.97      0.97      0.97       398
    Phishing       0.97      0.97      0.97       402

    accuracy                           0.97       800
   macro avg       0.97      0.97      0.97       800
weighted avg       0.97      0.97      0.97       800

[*] Confusion matrix image saved to phishing_model_confusion_matrix.png
[*] Trained model saved to phishing_model.pkl

------------------------------------------------------------
SAMPLE CLASSIFICATIONS
------------------------------------------------------------
[Phishing (94.2%)] Dear user, your account has been suspended. Click here immediately to verify: ht...
[Safe (98.1%)] Hi Sarah, attached is the agenda for tomorrow's meeting. Let me know if you'd li...
```

(Exact numbers vary run to run depending on the data split and dataset version. On the synthetic fallback dataset, accuracy is often 100% since its templates are highly distinct — real-world accuracy on the downloaded dataset typically falls in the 93–98% range.)

## Expected Outcome

The model successfully classifies emails as **Phishing** or **Safe** with high accuracy, based on a combination of textual content and URL/keyword-based features — matching the goal of the original task.

## License

This project is provided for educational use. Add a license of your choice (e.g. MIT) if publishing publicly.
