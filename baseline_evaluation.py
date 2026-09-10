import pickle
import re
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# LOAD MODEL
# ============================================================

rf = pickle.load(
    open("random_forest_model.pkl", "rb")
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

TRUSTED_DOMAINS = [
    'google.com', 'facebook.com', 'youtube.com', 'twitter.com',
    'linkedin.com', 'instagram.com', 'microsoft.com', 'apple.com',
    'amazon.com', 'netflix.com', 'github.com', 'wikipedia.org',
    'reddit.com', 'whatsapp.com', 'zoom.us', 'dropbox.com',
    'adobe.com', 'spotify.com', 'paypal.com', 'ebay.com',
    'yahoo.com', 'bing.com', 'office.com', 'live.com',
    'outlook.com', 'gmail.com', 'icloud.com', 'stackoverflow.com'
]


FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at",
    "num_question",
    "num_equal",
    "num_digits",
    "has_https",
    "has_http",
    "has_ip",
    "num_subdomains",
    "url_depth",
    "has_suspicious_words",
    "is_trusted_domain"
]


def extract_features(url):

    url = str(url).strip()

    try:
        parsed = urlparse(
            url if url.startswith("http")
            else "http://" + url
        )

        domain = parsed.netloc.lower().replace("www.", "")

    except Exception:
        domain = ""

    is_trusted = int(
        any(
            domain == td or domain.endswith("." + td)
            for td in TRUSTED_DOMAINS
        )
    )

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('_'),
        url.count('/'),
        url.count('@'),
        url.count('?'),
        url.count('='),
        sum(c.isdigit() for c in url),
        int(url.startswith('https')),
        int(url.startswith('http')),
        int(bool(
            re.search(
                r'\d+\.\d+\.\d+\.\d+',
                url
            )
        )),
        max(url.count('.') - 1, 0),
        url.count('/'),
        int(any(
            word in url.lower()
            for word in [
                'login',
                'verify',
                'secure',
                'account',
                'update',
                'banking',
                'confirm',
                'paypal',
                'ebay'
            ]
        )),
        is_trusted
    ]


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("phishing_site_urls.csv")

df.columns = (
    df.columns
    .astype(str)
    .str.replace('\ufeff', '', regex=False)
    .str.strip()
    .str.lower()
)

print("Columns:", df.columns.tolist())


# ============================================================
# CLEAN
# ============================================================

df = df[["url", "label"]].dropna()

df["url"] = df["url"].astype(str).str.strip()


# ============================================================
# LABEL CONVERSION
# ============================================================

df["target"] = df["label"].map({
    "good": 0,
    "bad": 1
})

df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


print("\nDataset size:", len(df))
print("\nClass distribution:")
print(df["target"].value_counts())


# ============================================================
# RANDOM TRAIN/TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["target"]
)


print("\n==============================")
print("RANDOM SPLIT")
print("==============================")

print("Training URLs:", len(train_df))
print("Testing URLs :", len(test_df))


# ============================================================
# TEST FEATURES
# ============================================================

X_test = pd.DataFrame(
    [
        extract_features(url)
        for url in test_df["url"]
    ],
    columns=FEATURE_NAMES
)

y_test = test_df["target"].values


# ============================================================
# PREDICTION
# ============================================================

predictions = rf.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n==============================")
print("RANDOM SPLIT RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ============================================================
# REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Safe",
            "Phishing"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE
# ============================================================

results = pd.DataFrame({
    "metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],
    "Random Split": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    "baseline_results.csv",
    index=False
)

print(
    "\n✓ Results saved to baseline_results.csv"
)