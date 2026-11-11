"""
Training and evaluation entry point for Insurance Fraud Detection.

Run from the project root:
    python train.py

Expected processed files:
    Data/Processed/X_train.npz
    Data/Processed/X_test.npz
    Data/Processed/y_train.npy
    Data/Processed/y_test.npy
    Data/Processed/groups_train.csv
    Data/Processed/groups_test.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model import (
    GroupAwareStackingClassifier,
    create_random_forest,
    create_voting_ensemble,
    create_xgboost,
)


# ==================================================
# 1. CONFIGURATION
# ==================================================

RANDOM_STATE = 42
STACKING_FOLDS = 5
CLASSIFICATION_THRESHOLD = 0.50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD PREPROCESSED DATA
# ==================================================

def load_data():
    """Load sparse feature matrices, labels, and customer groups."""

    required_files = [
        "X_train.npz",
        "X_test.npz",
        "y_train.npy",
        "y_test.npy",
        "groups_train.csv",
        "groups_test.csv",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (PROCESSED_DIR / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing processed data files:\n"
            + "\n".join(missing_files)
            + "\n\nRun all preprocessing.ipynb cells, including the save cell."
        )

    X_train = load_npz(PROCESSED_DIR / "X_train.npz").tocsr()
    X_test = load_npz(PROCESSED_DIR / "X_test.npz").tocsr()

    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")

    # The notebook saves the Series index as a CSV header.
    groups_train = pd.read_csv(
        PROCESSED_DIR / "groups_train.csv"
    ).iloc[:, 0].to_numpy()

    groups_test = pd.read_csv(
        PROCESSED_DIR / "groups_test.csv"
    ).iloc[:, 0].to_numpy()

    if not (
        X_train.shape[0] == len(y_train) == len(groups_train)
    ):
        raise ValueError("Training feature, target, and group lengths differ.")

    if not (
        X_test.shape[0] == len(y_test) == len(groups_test)
    ):
        raise ValueError("Testing feature, target, and group lengths differ.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and testing feature counts differ.")

    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise ValueError(
            f"Customer leakage detected: {len(overlap)} groups overlap."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    )


# ==================================================
# 3. EVALUATION
# ==================================================

def evaluate_model(name, model, X_test, y_test):
    """Evaluate a fitted classifier and return metrics."""

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(np.int8)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "PR_AUC": average_precision_score(y_test, probabilities),
    }

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)
    print(pd.Series(metrics).to_string(index=False))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Legitimate", "Fraudulent"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return metrics


# ==================================================
# 4. TRAIN MODELS
# ==================================================

def main():
    print("=" * 65)
    print("INSURANCE FRAUD DETECTION - MODEL TRAINING")
    print("=" * 65)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        groups_test,
    ) = load_data()

    print("\nData loaded successfully.")
    print("Training matrix:", X_train.shape)
    print("Testing matrix:", X_test.shape)
    print("Training fraud rate:", round(y_train.mean() * 100, 2), "%")
    print("Testing fraud rate:", round(y_test.mean() * 100, 2), "%")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    print("XGBoost scale_pos_weight:", round(scale_pos_weight, 3))

    random_forest = create_random_forest(RANDOM_STATE)
    xgboost = create_xgboost(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    results = []
    trained_models = {}

    # ----------------------------------------------
    # Model 1: Random Forest
    # ----------------------------------------------
    print("\n[1/4] Training Random Forest...")
    rf_model = random_forest
    rf_model.fit(X_train, y_train)

    trained_models["random_forest"] = rf_model
    results.append(
        evaluate_model(
            "Random Forest",
            rf_model,
            X_test,
            y_test,
        )
    )

    # ----------------------------------------------
    # Model 2: XGBoost
    # ----------------------------------------------
    print("\n[2/4] Training XGBoost...")
    xgb_model = xgboost
    xgb_model.fit(X_train, y_train)

    trained_models["xgboost"] = xgb_model
    results.append(
        evaluate_model(
            "XGBoost",
            xgb_model,
            X_test,
            y_test,
        )
    )

    # ----------------------------------------------
    # Model 3: Soft Voting
    # ----------------------------------------------
    print("\n[3/4] Training Soft Voting Ensemble...")
    voting_model = create_voting_ensemble(
        create_random_forest(RANDOM_STATE),
        create_xgboost(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        ),
    )
    voting_model.fit(X_train, y_train)

    trained_models["soft_voting"] = voting_model
    results.append(
        evaluate_model(
            "Soft Voting",
            voting_model,
            X_test,
            y_test,
        )
    )

    # ----------------------------------------------
    # Model 4: Group-aware Stacking
    # ----------------------------------------------
    print("\n[4/4] Training Group-aware Stacking Ensemble...")
    stacking_model = GroupAwareStackingClassifier(
        random_forest=create_random_forest(RANDOM_STATE),
        xgboost=create_xgboost(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
        ),
        n_splits=STACKING_FOLDS,
        random_state=RANDOM_STATE,
    )

    stacking_model.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    trained_models["stacking"] = stacking_model
    results.append(
        evaluate_model(
            "Group-aware Stacking",
            stacking_model,
            X_test,
            y_test,
        )
    )

    # ==================================================
    # 5. SAVE MODELS AND RESULTS
    # ==================================================

    print("\nSaving trained models...")

    for name, model in trained_models.items():
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        print("Saved:", model_path)

    results_df = pd.DataFrame(results)
    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print("\nModel comparison:")
    print(
        results_df.sort_values(
            by="PR_AUC",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nResults saved to:", results_path)
    print("\nTraining completed.")


if __name__ == "__main__":
    main()
