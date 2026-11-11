
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data" / "Processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load test data and trained Random Forest.
X_test = load_npz(DATA_DIR / "X_test.npz").astype(np.float32)
y_test = np.load(DATA_DIR / "y_test.npy")

model = joblib.load(
    MODEL_DIR / "random_forest_smote_gridsearch.joblib"
)

print("Generating fraud probabilities...")

probabilities = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.05, 0.51, 0.025)

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "Predicted_Fraud": tp + fp,
    })


results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "threshold_analysis.csv",
    index=False,
)

print("\nTHRESHOLD ANALYSIS")
print("=" * 90)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nBest F1 threshold:")

best = results_df.loc[
    results_df["F1"].idxmax()
]

print(best.to_string())

print("\nResults saved to:", REPORT_DIR)
