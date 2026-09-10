import pandas as pd
import numpy as np
import re
import pickle
from urllib.parse import urlparse

from sklearn.metrics import accuracy_score


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
    "google.com", "facebook.com", "youtube.com",
    "twitter.com", "linkedin.com", "instagram.com",
    "microsoft.com", "apple.com", "amazon.com",
    "netflix.com", "github.com", "wikipedia.org",
    "reddit.com", "whatsapp.com", "zoom.us",
    "dropbox.com", "adobe.com", "spotify.com",
    "paypal.com", "ebay.com", "yahoo.com",
    "bing.com", "office.com", "live.com",
    "outlook.com", "gmail.com", "icloud.com",
    "stackoverflow.com"
}

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account",
    "update", "banking", "confirm", "paypal", "ebay"
]


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
        any(domain.endswith("." + td)
            for td in TRUSTED_DOMAINS)
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
        int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),
        trusted
    ]


def make_features(urls):

    return pd.DataFrame(
        [extract_features(url) for url in urls],
        columns=FEATURE_NAMES
    )


def perturb_url(url, method):

    url = str(url)

    if method == "add_query":
        if "?" in url:
            return url + "&ref=123"
        return url + "?ref=123"

    if method == "add_path":
        return url.rstrip("/") + "/home"

    if method == "add_digit":
        return url + "1"

    if method == "add_hyphen":
        return url.replace(".", "-.", 1)

    if method == "uppercase":
        return url.upper()

    if method == "add_www":
        if "://" in url:
            scheme, rest = url.split("://", 1)

            if not rest.startswith("www."):
                return scheme + "://www." + rest

        return url

    return url


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

with open("research_unseen_domain_rf.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("phishing_site_urls.csv")

df.columns = (
    df.columns
    .astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
    .str.lower()
)

df["label"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .map({"good": 0, "bad": 1})
)

df = df.dropna(subset=["url", "label"])

df["label"] = df["label"].astype(int)


# ---------------------------------------------------------
# DOMAIN SPLIT — SAME SEED AS MAIN EXPERIMENT
# ---------------------------------------------------------

def get_domain(url):

    try:
        if not str(url).startswith(("http://", "https://")):
            url = "http://" + str(url)

        hostname = urlparse(url).hostname

        if hostname:
            return hostname.lower().replace("www.", "")

    except:
        pass

    return ""


df["domain"] = df["url"].apply(get_domain)

domains = df["domain"].drop_duplicates().to_numpy()

from sklearn.model_selection import train_test_split

train_domains, test_domains = train_test_split(
    domains,
    test_size=0.20,
    random_state=42
)

test_domain_set = set(test_domains)

test_df = df[
    df["domain"].isin(test_domain_set)
].copy()


# ---------------------------------------------------------
# SAMPLE
# ---------------------------------------------------------

sample_size = min(1000, len(test_df))

sample = test_df.sample(
    n=sample_size,
    random_state=42
)

urls = sample["url"].tolist()
true_labels = sample["label"].to_numpy()


# ---------------------------------------------------------
# ORIGINAL PREDICTIONS
# ---------------------------------------------------------

X_original = make_features(urls)

original_predictions = model.predict(X_original)


# ---------------------------------------------------------
# ROBUSTNESS
# ---------------------------------------------------------

methods = [
    "add_query",
    "add_path",
    "add_digit",
    "add_hyphen",
    "uppercase",
    "add_www"
]

results = []


for method in methods:

    modified_urls = [
        perturb_url(url, method)
        for url in urls
    ]

    X_modified = make_features(modified_urls)

    modified_predictions = model.predict(X_modified)

    safe_to_phishing = np.sum(
        (original_predictions == 0) &
        (modified_predictions == 1)
    )

    phishing_to_safe = np.sum(
        (original_predictions == 1) &
        (modified_predictions == 0)
    )

    total_flips = safe_to_phishing + phishing_to_safe

    results.append({
        "Perturbation": method,
        "Samples": sample_size,
        "Safe_to_Phishing": int(safe_to_phishing),
        "Phishing_to_Safe": int(phishing_to_safe),
        "Total_Flips": int(total_flips),
        "Flip_Rate": total_flips / sample_size,
        "Stability": 1 - (total_flips / sample_size)
    })


results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("DETAILED ROBUSTNESS RESULTS")
print("=" * 75)

print(results_df.to_string(index=False))

results_df.to_csv(
    "robustness_detailed_results.csv",
    index=False
)

print("\nSaved:")
print("robustness_detailed_results.csv")