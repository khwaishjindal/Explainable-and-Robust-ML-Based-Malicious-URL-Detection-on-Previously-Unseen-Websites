import pickle
import re
from urllib.parse import urlparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD MODEL
# ============================================================

rf = pickle.load(open("random_forest_model.pkl", "rb"))


# ============================================================
# 2. FEATURE EXTRACTION
#    Must match the features used by the existing model
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
        int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url))),
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
# 3. LOAD DATASET
# ============================================================

DATASET = "phishing_site_urls.csv"

df = pd.read_csv(DATASET)

# Clean column names
df.columns = (
    df.columns
    .astype(str)
    .str.replace('\ufeff', '', regex=False)
    .str.strip()
    .str.lower()
)

print("\nColumns found:")
print(df.columns.tolist())


# ============================================================
# 4. FIND URL COLUMN
# ============================================================

url_column = next(
    (c for c in df.columns if "url" in c),
    None
)

if url_column is None:
    raise ValueError(
        "URL column not found. Columns are: "
        + str(df.columns.tolist())
    )


# ============================================================
# 5. FIND LABEL COLUMN
# ============================================================

label_column = next(
    (
        c for c in df.columns
        if c in ["label", "type", "class", "target"]
    ),
    None
)

if label_column is None:
    raise ValueError(
        "Label column not found. Columns are: "
        + str(df.columns.tolist())
    )


print("\nURL column:", url_column)
print("Label column:", label_column)


# ============================================================
# 6. CLEAN DATA
# ============================================================

df = df[[url_column, label_column]].dropna()

df[url_column] = df[url_column].astype(str).str.strip()

print("\nTotal URLs:", len(df))


# ============================================================
# 7. CONVERT LABELS
# ============================================================

print("\nOriginal labels:")
print(df[label_column].value_counts())


def convert_label(label):

    label = str(label).strip().lower()

    # Adjust according to common phishing dataset labels
    if label in ["bad", "phishing", "malicious", "malware", "1"]:
        return 1

    if label in ["good", "legitimate", "benign", "safe", "0"]:
        return 0

    return None


df["target"] = df[label_column].apply(convert_label)

df = df.dropna(subset=["target"])

df["target"] = df["target"].astype(int)


# ============================================================
# 8. EXTRACT DOMAINS
# ============================================================

def extract_domain(url):

    try:

        parsed = urlparse(
            url if url.startswith("http")
            else "http://" + url
        )

        return parsed.netloc.lower().replace("www.", "")

    except Exception:

        return ""


df["domain"] = df[url_column].apply(extract_domain)

df = df[df["domain"] != ""]


print("\nUnique domains:", df["domain"].nunique())


# ============================================================
# 9. DOMAIN-BASED SPLIT
# ============================================================

# Randomly select 20% of domains for testing
# These domains will NOT appear in training.

domains = df["domain"].drop_duplicates().to_numpy()

import numpy as np

np.random.seed(42)

np.random.shuffle(domains)

test_domain_count = int(len(domains) * 0.20)

test_domains = set(
    domains[:test_domain_count]
)

train_domains = set(
    domains[test_domain_count:]
)


train_df = df[
    df["domain"].isin(train_domains)
].copy()

test_df = df[
    df["domain"].isin(test_domains)
].copy()


print("\n==============================")
print("UNSEEN DOMAIN SPLIT")
print("==============================")

print("Training domains:", len(train_domains))
print("Testing domains :", len(test_domains))

print("Training URLs:", len(train_df))
print("Testing URLs :", len(test_df))


# ============================================================
# 10. VERIFY NO DOMAIN OVERLAP
# ============================================================

overlap = (
    set(train_df["domain"])
    &
    set(test_df["domain"])
)

print("\nDomain overlap:", len(overlap))

if len(overlap) > 0:

    raise ValueError(
        "ERROR: Training and testing domains overlap!"
    )

print("✓ No domain overlap")


# ============================================================
# 11. CREATE TEST FEATURES
# ============================================================

X_test = pd.DataFrame(
    [
        extract_features(url)
        for url in test_df[url_column]
    ],
    columns=[
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
)

y_test = test_df["target"].values


# ============================================================
# 12. PREDICT
# ============================================================

predictions = rf.predict(X_test)


# ============================================================
# 13. EVALUATION
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
print("UNSEEN DOMAIN RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Safe", "Phishing"],
        zero_division=0
    )
)


# ============================================================
# 16. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],

    "Random Forest": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

results.to_csv(
    "unseen_domain_results.csv",
    index=False
)

print("\n✓ Results saved to unseen_domain_results.csv")