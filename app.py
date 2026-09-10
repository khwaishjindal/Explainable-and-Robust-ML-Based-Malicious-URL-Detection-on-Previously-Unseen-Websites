from urllib.parse import urlparse
from textwrap import dedent
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import shap
import lime
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Phishing URL Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    *, html, body, [class*="css"], .stApp {
        background-color: #0f1117 !important;
        font-family: 'IBM Plex Sans', sans-serif;
        color: #e2e8f0;
    }

    .block-container {
        padding: 5rem 3rem 3rem !important;
        max-width: 960px !important;
    }

    /* Title */

    .title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.4rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 2rem;
        text-align: center;
    }

    .title span {
        color: #4ade80;
    }

    /* Input */

    div[data-testid="stTextInput"] input {
        background-color: #1e2433 !important;
        border: 1px solid #2d3748 !important;
        color: #e2e8f0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        border-radius: 6px !important;
        font-size: 1.1rem !important;
        padding: 1rem 1.2rem !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #4ade80 !important;
        box-shadow: none !important;
    }

    div[data-testid="stTextInput"] input::placeholder {
        color: #4a5568 !important;
    }

    /* Button */

    .stButton > button {
        background-color: transparent !important;
        color: #e2e8f0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        padding: 0.9rem !important;
        transition: all 0.15s !important;
    }

    .stButton > button:hover {
        border-color: #4ade80 !important;
        color: #4ade80 !important;
    }

    /* Result */

    .result-bad {
        background: #1a0e0e;
        border: 1px solid #fc8181;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    .result-good {
        background: #0e1a12;
        border: 1px solid #4ade80;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    .result-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
    }

    .result-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }

    /* Cards */

    .cards-row {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 1rem;
        margin-top: 3rem;
    }

    .card {
        background: #1e2433;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 2rem;
    }

    .card-icon {
        font-size: 2rem;
        margin-bottom: 0.8rem;
    }

    .card-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1rem;
        color: #4ade80;
        margin-bottom: 0.7rem;
        font-weight: 600;
    }

    .card-body {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.8;
    }

    /* Divider */

    .divider {
        border: none;
        border-top: 1px solid #2d3748;
        margin: 2rem 0;
    }

    /* XAI section */

    .xai-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #4ade80;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Tabs */

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid #2d3748 !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        color: #64748b !important;
        background: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: #4ade80 !important;
    }

    footer,
    #MainMenu,
    header {
        visibility: hidden;
    }

    @media (max-width: 768px) {

        .block-container {
            padding: 3rem 1rem 2rem !important;
        }

        .title {
            font-size: 1.5rem !important;
        }

        .cards-row {
            grid-template-columns: 1fr !important;
        }

        .card {
            padding: 1.2rem !important;
        }

        div[data-testid="stTextInput"] input {
            font-size: 0.9rem !important;
            padding: 0.7rem 0.8rem !important;
        }

        .stButton > button {
            font-size: 0.9rem !important;
            padding: 0.7rem !important;
        }

        .result-title {
            font-size: 1.1rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    rf = pickle.load(
        open("random_forest_model.pkl", "rb")
    )

    lr = pickle.load(
        open("logistic_regression_model.pkl", "rb")
    )

    scaler = pickle.load(
        open("scaler.pkl", "rb")
    )

    feature_names = pickle.load(
        open("feature_names.pkl", "rb")
    )

    return rf, lr, scaler, feature_names


rf, lr, scaler, feature_names = load_models()


# ============================================================
# TRUSTED DOMAINS
# ============================================================

TRUSTED_DOMAINS = [
    "google.com",
    "facebook.com",
    "youtube.com",
    "twitter.com",
    "linkedin.com",
    "instagram.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "netflix.com",
    "github.com",
    "wikipedia.org",
    "reddit.com",
    "whatsapp.com",
    "zoom.us",
    "dropbox.com",
    "adobe.com",
    "spotify.com",
    "paypal.com",
    "ebay.com",
    "yahoo.com",
    "bing.com",
    "office.com",
    "live.com",
    "outlook.com",
    "gmail.com",
    "icloud.com",
    "stackoverflow.com"
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(url):

    try:

        parsed = urlparse(
            url if url.startswith("http") else "http://" + url
        )

        domain = (
            parsed.netloc
            .lower()
            .replace("www.", "")
        )

    except Exception:

        domain = ""

    is_trusted = 1 if any(
        domain == td or domain.endswith("." + td)
        for td in TRUSTED_DOMAINS
    ) else 0

    features = {

        "url_length": len(url),

        "num_dots": url.count("."),

        "num_hyphens": url.count("-"),

        "num_underscores": url.count("_"),

        "num_slashes": url.count("/"),

        "num_at": url.count("@"),

        "num_question": url.count("?"),

        "num_equal": url.count("="),

        "num_digits": sum(
            c.isdigit()
            for c in url
        ),

        "has_https": 1 if url.startswith("https") else 0,

        "has_http": 1 if url.startswith("http") else 0,

        "has_ip": 1 if re.search(
            r"\d+\.\d+\.\d+\.\d+",
            url
        ) else 0,

        "num_subdomains": url.count(".") - 1,

        "url_depth": url.count("/"),

        "has_suspicious_words": 1 if any(
            word in url.lower()
            for word in [
                "login",
                "verify",
                "secure",
                "account",
                "update",
                "banking",
                "confirm",
                "paypal",
                "ebay"
            ]
        ) else 0,

        "is_trusted_domain": is_trusted
    }

    return pd.DataFrame([features])


# ============================================================
# XAI BACKGROUND DATA
# ============================================================

@st.cache_data
def load_xai_background():

    dataset = pd.read_csv(
        "phishing_site_urls.csv"
    )

    # Clean column names
    dataset.columns = (
        dataset.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    url_columns = [
        column
        for column in dataset.columns
        if "url" in column
    ]

    if not url_columns:
        raise ValueError(
            "URL column was not found in the dataset."
        )

    url_column = url_columns[0]

    urls = (
        dataset[url_column]
        .dropna()
        .astype(str)
    )

    sample_size = min(
        500,
        len(urls)
    )

    urls = urls.sample(
        sample_size,
        random_state=42
    )

    background = pd.concat(
        [
            extract_features(url)
            for url in urls
        ],
        ignore_index=True
    )

    return background


xai_background = load_xai_background()


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    """
    <div class="title">
        🛡️ Phishing URL <span>Detector</span>
    </div>
    """,
    unsafe_allow_html=True
)


# URL INPUT

_, center, _ = st.columns([1, 4, 1])

with center:

    url_input = st.text_input(
        "",
        placeholder="Paste a URL here — e.g. https://paypal-login.verify.com",
        label_visibility="collapsed"
    )


# CHECK BUTTON

_, btn_col, _ = st.columns([2, 1, 2])

with btn_col:

    analyze = st.button(
        "Check URL"
    )

# ============================================================
# INFORMATION CARDS
# ============================================================

card1, card2, card3 = st.columns(3)

with card1:
    st.markdown("### 🔍 Layer 1 — Detection")
    st.write(
        "Random Forest acts as the primary classifier, "
        "with Logistic Regression used as a baseline."
    )

with card2:
    st.markdown("### 🧠 Layer 2 — Explainability")
    st.write(
        "SHAP and LIME provide explanations for "
        "individual URL predictions."
    )

with card3:
    st.markdown("### ⚡ How It Works")
    st.write(
        "16 features are extracted from the URL structure, "
        "including length, dots, hyphens, keywords, "
        "IP presence, and more."
    )



# ============================================================
# URL ANALYSIS
# ============================================================

if analyze and url_input.strip():

    url = url_input.strip()

    # Extract the 16 URL features
    features_df = extract_features(url)

    # Scale features for Logistic Regression baseline
    scaled = scaler.transform(features_df)

    # Random Forest prediction
    rf_pred = rf.predict(features_df)[0]

    rf_prob = rf.predict_proba(features_df)[0]

    # Logistic Regression baseline prediction
    lr_pred = lr.predict(scaled)[0]

    lr_prob = lr.predict_proba(scaled)[0]

    st.markdown(
        '<hr class="divider">',
        unsafe_allow_html=True
    )


    # ========================================================
    # RESULT
    # ========================================================

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        if rf_pred == 1:

         st.markdown(
        dedent(f"""
        <div class="result-bad">
            <div class="result-title" style="color:#fc8181">
                ⚠ Phishing
            </div>
            <div class="result-sub">
                Random Forest · {rf_prob[1] * 100:.1f}% confidence
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

        else:

         st.markdown(
        dedent(f"""
        <div class="result-good">
            <div class="result-title" style="color:#4ade80">
                ✓ Safe
            </div>
            <div class="result-sub">
                Random Forest · {rf_prob[0] * 100:.1f}% confidence
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )


    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown(
        '<hr class="divider">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="xai-label">// Explainability</div>',
        unsafe_allow_html=True
    )

    # IMPORTANT:
    # Both tabs are created inside the analysis block.
    tab_shap, tab_lime = st.tabs(
        ["SHAP", "LIME"]
    )


    # ========================================================
    # SHAP
    # ========================================================

    with tab_shap:

        try:

            # Explain the Random Forest classifier
            explainer = shap.TreeExplainer(rf)

            shap_explanation = explainer(
                features_df
            )

            shap_values = shap_explanation.values

            # Handle SHAP output shapes across versions
            if shap_values.ndim == 3:

                # Binary classification:
                # class 1 = Phishing
                shap_vals = shap_values[
                    0, :, 1
                ]

            elif shap_values.ndim == 2:

                shap_vals = shap_values[0]

            else:

                shap_vals = shap_values


            # Make sure feature names match
            if len(feature_names) != len(shap_vals):

                feature_names_plot = list(
                    features_df.columns
                )

            else:

                feature_names_plot = list(
                    feature_names
                )


            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            fig.patch.set_facecolor(
                "#0f1117"
            )

            ax.set_facecolor(
                "#1e2433"
            )


            colors = [
                "#fc8181" if value > 0
                else "#4ade80"
                for value in shap_vals
            ]


            ax.barh(
                feature_names_plot,
                shap_vals,
                color=colors,
                edgecolor="none",
                height=0.5
            )


            ax.set_xlabel(
                "SHAP Value",
                color="#64748b",
                fontsize=8
            )


            ax.tick_params(
                colors="#94a3b8",
                labelsize=7.5
            )


            for spine in ax.spines.values():

                spine.set_color(
                    "#2d3748"
                )


            ax.axvline(
                0,
                color="#2d3748",
                linewidth=1
            )


            ax.set_title(
                "Which features mattered most?",
                color="#e2e8f0",
                fontsize=9,
                pad=10
            )


            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        except Exception as e:

            st.error(
                f"SHAP error: {e}"
            )


    # ========================================================
    # LIME
    # ========================================================

    with tab_lime:

        try:

            bg = xai_background.values


            lime_exp = (
                lime.lime_tabular
                .LimeTabularExplainer(
                    training_data=bg,
                    feature_names=feature_names,
                    class_names=[
                        "Safe",
                        "Phishing"
                    ],
                    mode="classification",
                    random_state=42
                )
            )


            exp = lime_exp.explain_instance(
                features_df.values[0],
                rf.predict_proba,
                num_features=10
            )


            lime_list = exp.as_list()


            lime_feats = [
                item[0]
                for item in lime_list
            ]


            lime_weights = [
                item[1]
                for item in lime_list
            ]


            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            fig.patch.set_facecolor(
                "#0f1117"
            )

            ax.set_facecolor(
                "#1e2433"
            )


            colors = [
                "#fc8181" if weight > 0
                else "#4ade80"
                for weight in lime_weights
            ]


            ax.barh(
                lime_feats,
                lime_weights,
                color=colors,
                edgecolor="none",
                height=0.5
            )


            ax.set_xlabel(
                "Impact",
                color="#64748b",
                fontsize=8
            )


            ax.tick_params(
                colors="#94a3b8",
                labelsize=7.5
            )


            for spine in ax.spines.values():

                spine.set_color(
                    "#2d3748"
                )


            ax.axvline(
                0,
                color="#2d3748",
                linewidth=1
            )


            ax.set_title(
                "Why did the model decide this?",
                color="#e2e8f0",
                fontsize=9,
                pad=10
            )


            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        except Exception as e:

            st.error(
                f"LIME error: {e}"
            )


# ============================================================
# EMPTY INPUT WARNING
# ============================================================

if analyze and not url_input.strip():

    st.warning(
        "Please paste a URL first."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "<br><br>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="
        text-align:center;
        color:#2d3748;
        font-size:0.75rem;
        font-family:IBM Plex Mono, monospace;
    ">
        Built with Random Forest · SHAP · LIME
        <br>By: Khwaish , Vanshika , Aanchal , Ishika
    </p>
    """,
    unsafe_allow_html=True
)