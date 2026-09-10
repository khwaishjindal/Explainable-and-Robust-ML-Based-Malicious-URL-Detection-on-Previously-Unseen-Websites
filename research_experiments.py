import pandas as pd
import numpy as np
import re
import pickle
import time
from urllib.parse import urlparse

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

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

TRUSTED_DOMAINS = {
    "google.com", "facebook.com", "youtube.com", "twitter.com",
    "linkedin.com", "instagram.com", "microsoft.com", "apple.com",
    "amazon.com", "netflix.com", "github.com", "wikipedia.org",
    "reddit.com", "whatsapp.com", "zoom.us", "dropbox.com",
    "adobe.com", "spotify.com", "paypal.com", "ebay.com",
    "yahoo.com", "bing.com", "office.com", "live.com",
    "outlook.com", "gmail.com", "icloud.com", "stackoverflow.com"
}

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account",
    "update", "banking", "confirm", "paypal", "ebay"
]


def get_domain(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        hostname = urlparse(url).hostname

        if hostname:
            return hostname.lower().replace("www.", "")

    except Exception:
        pass

    return ""


def extract_features(url):
    url = str(url)

    try:
        parsed = urlparse(
            url if url.startswith(("http://", "https://"))
            else "http://" + url
        )

        domain = parsed.hostname.lower() if parsed.hostname else ""
        domain = domain.replace("www.", "")

    except Exception:
        domain = ""

    trusted = int(
        domain in TRUSTED_DOMAINS or
        any(domain.endswith("." + td) for td in TRUSTED_DOMAINS)
    )

    return [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("_"),
        url.count("/"),
        url.count("@"),
        url.count("?"),
        url.count("="),
        sum(c.isdigit() for c in url),
        int(url.lower().startswith("https")),
        int(url.lower().startswith("http")),
        int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        max(url.count(".") - 1, 0),
        url.count("/"),
        int(any(word in url.lower() for word in SUSPICIOUS_WORDS)),
        trusted
    ]


def make_features(urls):
    return pd.DataFrame(
        [extract_features(url) for url in urls],
        columns=FEATURE_NAMES
    )


def evaluate_model(model, X_test, y_test, name):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Safe", "Phishing"],
            zero_division=0
        )
    )

    return {
        "Experiment": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# LOAD DATA
# ============================================================

start = time.time()

print("\nLoading dataset...")

df = pd.read_csv("phishing_site_urls.csv")

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)

print("Columns:", df.columns.tolist())
print("Dataset size:", len(df))

url_column = "url"
label_column = "label"

df = df.dropna(subset=[url_column, label_column]).copy()

df[label_column] = (
    df[label_column]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "good": 0,
        "bad": 1
    })
)

df = df.dropna(subset=[label_column])
df[label_column] = df[label_column].astype(int)

print("\nClass distribution:")
print(df[label_column].value_counts())


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\nExtracting URL features...")
print("This may take a few minutes...")

X_all = make_features(df[url_column].values)

y_all = df[label_column].values

print("Feature extraction completed.")
print("Feature matrix:", X_all.shape)


# ============================================================
# EXPERIMENT 1
# TRUE RANDOM TRAIN/TEST BASELINE
# ============================================================

print("\n\n")
print("#" * 70)
print("EXPERIMENT 1: RANDOM 80/20 BASELINE")
print("#" * 70)

X_train_random, X_test_random, y_train_random, y_test_random = train_test_split(
    X_all,
    y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all
)

print("Training URLs:", len(X_train_random))
print("Testing URLs :", len(X_test_random))

random_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining Random Forest...")
random_rf.fit(X_train_random, y_train_random)

random_result = evaluate_model(
    random_rf,
    X_test_random,
    y_test_random,
    "Random Split Baseline"
)

with open("research_random_rf.pkl", "wb") as f:
    pickle.dump(random_rf, f)


# ============================================================
# EXPERIMENT 2
# TRUE UNSEEN-DOMAIN EXPERIMENT
# ============================================================

print("\n\n")
print("#" * 70)
print("EXPERIMENT 2: PROPER UNSEEN-DOMAIN EVALUATION")
print("#" * 70)

print("\nExtracting domains...")

df["domain"] = df[url_column].apply(get_domain)

unique_domains = df["domain"].drop_duplicates().to_numpy()

print("Unique domains:", len(unique_domains))

train_domains, test_domains = train_test_split(
    unique_domains,
    test_size=0.20,
    random_state=42
)

train_domain_set = set(train_domains)
test_domain_set = set(test_domains)

train_mask = df["domain"].isin(train_domain_set)
test_mask = df["domain"].isin(test_domain_set)

domain_train = df[train_mask].copy()
domain_test = df[test_mask].copy()

print("\nTraining domains:", len(train_domains))
print("Testing domains :", len(test_domains))

print("Training URLs:", len(domain_train))
print("Testing URLs :", len(domain_test))

overlap = train_domain_set.intersection(test_domain_set)

print("Domain overlap:", len(overlap))

if len(overlap) != 0:
    raise ValueError("ERROR: Domain leakage detected!")

print("\nExtracting train/test features...")

X_domain_train = make_features(
    domain_train[url_column].values
)

X_domain_test = make_features(
    domain_test[url_column].values
)

y_domain_train = domain_train[label_column].values
y_domain_test = domain_test[label_column].values


unseen_rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

print("\nTraining NEW Random Forest ONLY on training domains...")

unseen_rf.fit(
    X_domain_train,
    y_domain_train
)

unseen_result = evaluate_model(
    unseen_rf,
    X_domain_test,
    y_domain_test,
    "Proper Unseen-Domain Evaluation"
)

with open("research_unseen_domain_rf.pkl", "wb") as f:
    pickle.dump(unseen_rf, f)


# ============================================================
# EXPERIMENT 3
# ROBUSTNESS TESTING
# ============================================================

print("\n\n")
print("#" * 70)
print("EXPERIMENT 3: ROBUSTNESS TESTING")
print("#" * 70)


def perturb_url(url, method):

    url = str(url)

    if method == "add_query":
        if "?" in url:
            return url + "&ref=123"
        return url + "?ref=123"

    elif method == "add_path":
        clean = url.rstrip("/")
        return clean + "/home"

    elif method == "add_digit":
        return url + "1"

    elif method == "add_hyphen":
        return url.replace(".", "-.", 1)

    elif method == "uppercase":
        return url.upper()

    elif method == "add_www":
        if "://" in url:
            scheme, rest = url.split("://", 1)

            if not rest.startswith("www."):
                return scheme + "://www." + rest

        return url

    return url


perturbations = [
    "add_query",
    "add_path",
    "add_digit",
    "add_hyphen",
    "uppercase",
    "add_www"
]


# Use a manageable sample from unseen-domain test data
sample_size = min(1000, len(domain_test))

robustness_sample = domain_test.sample(
    n=sample_size,
    random_state=42
).copy()

robustness_records = []

original_urls = robustness_sample[url_column].tolist()

X_original = make_features(original_urls)

original_predictions = unseen_rf.predict(X_original)

for method in perturbations:

    modified_urls = [
        perturb_url(url, method)
        for url in original_urls
    ]

    X_modified = make_features(modified_urls)

    modified_predictions = unseen_rf.predict(X_modified)

    flips = np.sum(
        original_predictions != modified_predictions
    )

    flip_rate = flips / len(original_predictions)

    stability = 1 - flip_rate

    print(
        f"{method:15s} | "
        f"Flip Rate: {flip_rate:.4f} | "
        f"Stability: {stability:.4f}"
    )

    robustness_records.append({
        "Perturbation": method,
        "Samples": len(original_predictions),
        "Prediction_Flips": int(flips),
        "Flip_Rate": flip_rate,
        "Prediction_Stability": stability
    })


robustness_df = pd.DataFrame(robustness_records)

robustness_df.to_csv(
    "robustness_results.csv",
    index=False
)

print("\nRobustness results saved to robustness_results.csv")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("#" * 70)
print("FINAL MODEL COMPARISON")
print("#" * 70)

results_df = pd.DataFrame([
    random_result,
    unseen_result
])

results_df.to_csv(
    "final_experiment_results.csv",
    index=False
)

print("\n")
print(results_df.to_string(index=False))

print("\nResults saved to:")
print("  final_experiment_results.csv")
print("  robustness_results.csv")
print("  research_random_rf.pkl")
print("  research_unseen_domain_rf.pkl")

print("\nTotal execution time:")
print(f"{(time.time() - start) / 60:.2f} minutes")

print("\nALL EXPERIMENTS COMPLETED.")