import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv("final_experiment_results.csv")
robustness = pd.read_csv("robustness_detailed_results.csv")


# ============================================================
# FIGURE 1 — MODEL PERFORMANCE
# ============================================================

metrics = ["Accuracy", "Precision", "Recall", "F1"]

random_values = results.iloc[0][metrics].values
unseen_values = results.iloc[1][metrics].values

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    random_values,
    width,
    label="Random Split"
)

plt.bar(
    x + width / 2,
    unseen_values,
    width,
    label="Unseen Domain"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Performance Comparison: Random vs Unseen-Domain Evaluation")
plt.legend()
plt.tight_layout()

plt.savefig(
    "performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2 — F1 DROP
# ============================================================

plt.figure(figsize=(7, 5))

values = [
    results.iloc[0]["F1"],
    results.iloc[1]["F1"]
]

plt.bar(
    ["Random Split", "Unseen Domain"],
    values
)

plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.title("F1-score Under Different Evaluation Strategies")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3 — ROBUSTNESS
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    robustness["Perturbation"],
    robustness["Flip_Rate"]
)

plt.ylabel("Prediction Flip Rate")
plt.xlabel("URL Perturbation")
plt.ylim(0, 0.4)
plt.title("Robustness Against Controlled URL Perturbations")

plt.xticks(rotation=30)

for i, value in enumerate(robustness["Flip_Rate"]):
    plt.text(
        i,
        value + 0.01,
        f"{value:.1%}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "robustness_flip_rate.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4 — SAFE → PHISHING VS PHISHING → SAFE
# ============================================================

x = np.arange(len(robustness))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    robustness["Safe_to_Phishing"],
    width,
    label="Safe → Phishing"
)

plt.bar(
    x + width / 2,
    robustness["Phishing_to_Safe"],
    width,
    label="Phishing → Safe"
)

plt.xticks(
    x,
    robustness["Perturbation"],
    rotation=30
)

plt.ylabel("Number of Prediction Flips")
plt.title("Direction of Prediction Changes Under Perturbations")
plt.legend()

plt.tight_layout()

plt.savefig(
    "robustness_flip_direction.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("✓ All figures generated successfully.")

print("\nGenerated files:")
print("1. performance_comparison.png")
print("2. f1_comparison.png")
print("3. robustness_flip_rate.png")
print("4. robustness_flip_direction.png")