
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
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


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
SMOTE_NEIGHBORS = 5

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "Data" / "Processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading processed data...")

    X_train = load_npz(DATA_DIR / "X_train.npz").astype(np.float32)
    X_test = load_npz(DATA_DIR / "X_test.npz").astype(np.float32)

    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    groups_train = pd.read_csv(
        DATA_DIR / "groups_train.csv"
    ).iloc[:, 0].to_numpy()

    groups_test = pd.read_csv(
        DATA_DIR / "groups_test.csv"
    ).iloc[:, 0].to_numpy()

    if set(groups_train) & set(groups_test):
        raise ValueError("Customer overlap between training and test data.")

    with open(
        DATA_DIR / "feature_names.json",
        "r",
        encoding="utf-8",
    ) as file:
        feature_names = json.load(file)

    # Support either a list or a dictionary containing feature names.
    if isinstance(feature_names, dict):
        for key in ("feature_names", "features", "columns"):
            if key in feature_names:
                feature_names = feature_names[key]
                break

    feature_names = list(feature_names)

    if len(feature_names) != X_train.shape[1]:
        print("Warning: Feature-name count does not match matrix.")
        print("Using generated feature names.")

        feature_names = [
            f"feature_{i}" for i in range(X_train.shape[1])
        ]

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        feature_names,
    )


# ============================================================
# 2. LOAD BEST RANDOM FOREST PARAMETERS
# ============================================================

def load_best_parameters():

    path = REPORT_DIR / "random_forest_best_params.json"

    with open(path, "r", encoding="utf-8") as file:
        params = json.load(file)

    print("\nUsing previously selected Random Forest parameters:")
    print(params)

    return params


# ============================================================
# 3. CREATE MODEL PIPELINE
# ============================================================

def create_pipeline():

    return Pipeline([
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=SMOTE_NEIGHBORS,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
    ])


# ============================================================
# 4. CREATE GROUP-AWARE VALIDATION SPLIT
# ============================================================

def create_validation_split(X, y, groups):

    print("\nCreating group-aware validation split...")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_indices, validation_indices = next(
        cv.split(X, y, groups)
    )

    X_subtrain = X[train_indices]
    X_validation = X[validation_indices]

    y_subtrain = y[train_indices]
    y_validation = y[validation_indices]

    groups_subtrain = groups[train_indices]
    groups_validation = groups[validation_indices]

    overlap = (
        set(groups_subtrain) & set(groups_validation)
    )

    if overlap:
        raise ValueError("Customer leakage in validation split.")

    print(f"Subtraining samples: {len(train_indices)}")
    print(f"Validation samples: {len(validation_indices)}")
    print(f"Customer overlap: {len(overlap)}")

    print("\nValidation class distribution:")
    print(pd.Series(y_validation).value_counts().sort_index())

    return (
        X_subtrain,
        X_validation,
        y_subtrain,
        y_validation,
    )


# ============================================================
# 5. TRAIN VALIDATION MODEL
# ============================================================

def train_validation_model(
    X_subtrain,
    y_subtrain,
    best_params,
):

    print("\nTraining validation model...")

    model = create_pipeline()

    model.set_params(**best_params)

    model.fit(X_subtrain, y_subtrain)

    print("Validation model training completed.")

    return model


# ============================================================
# 6. SELECT THRESHOLD USING VALIDATION DATA
# ============================================================

def select_threshold(model, X_validation, y_validation):

    print("\nGenerating validation probabilities...")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    thresholds = np.arange(0.05, 0.501, 0.005)

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "True_Positives": tp,
            "False_Positives": fp,
            "False_Negatives": fn,
            "Predicted_Fraud": tp + fp,
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        REPORT_DIR / "validation_threshold_analysis.csv",
        index=False,
    )

    # Maximize F1. Break ties by preferring higher precision.
    best = results_df.sort_values(
        ["F1", "Precision", "Threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    selected_threshold = float(best["Threshold"])

    print("\nBest validation threshold:")
    print(best.to_string())

    print(
        f"\nSelected threshold: {selected_threshold:.3f}"
    )

    with open(
        REPORT_DIR / "selected_threshold.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump({
            "threshold": selected_threshold,
            "selection_method": "Maximum validation F1",
            "validation_precision": float(best["Precision"]),
            "validation_recall": float(best["Recall"]),
            "validation_f1": float(best["F1"]),
        }, file, indent=4)

    return selected_threshold, results_df


# ============================================================
# 7. REFIT MODEL ON ALL TRAINING DATA
# ============================================================

def train_final_model(
    X_train,
    y_train,
    best_params,
):

    print("\nRefitting Random Forest on all training data...")

    model = create_pipeline()

    model.set_params(**best_params)

    model.fit(X_train, y_train)

    model_path = (
        MODEL_DIR / "final_random_forest_fraud_model.joblib"
    )

    joblib.dump(model, model_path)

    print(f"Final model saved: {model_path}")

    return model


# ============================================================
# 8. FINAL EVALUATION ON UNTOUCHED TEST DATA
# ============================================================

def evaluate_final_model(
    model,
    X_test,
    y_test,
    threshold,
):

    print("\n")
    print("=" * 65)
    print("FINAL TEST SET EVALUATION")
    print("=" * 65)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "Threshold": threshold,
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
        "PR_AUC": average_precision_score(
            y_test, probabilities
        ),
    }

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not Fraud", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nFinal metrics:")

    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")

    results = pd.DataFrame([metrics])

    results.to_csv(
        REPORT_DIR / "final_test_evaluation.csv",
        index=False,
    )

    return metrics


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(model, feature_names):

    print("\nExtracting Random Forest feature importance...")

    classifier = model.named_steps["classifier"]

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)

    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTop 20 important features:")
    print(
        importance_df.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return importance_df


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("FRAUD MODEL VALIDATION AND FEATURE IMPORTANCE")
    print("=" * 65)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
        feature_names,
    ) = load_data()

    best_params = load_best_parameters()

    # Split training data into subtraining and validation.
    (
        X_subtrain,
        X_validation,
        y_subtrain,
        y_validation,
    ) = create_validation_split(
        X_train,
        y_train,
        groups_train,
    )

    # Train only on the subtraining partition.
    validation_model = train_validation_model(
        X_subtrain,
        y_subtrain,
        best_params,
    )

    # Select operating threshold without touching test data.
    threshold, _ = select_threshold(
        validation_model,
        X_validation,
        y_validation,
    )

    # Refit using all available training records.
    final_model = train_final_model(
        X_train,
        y_train,
        best_params,
    )

    # Evaluate once on the untouched test set.
    evaluate_final_model(
        final_model,
        X_test,
        y_test,
        threshold,
    )

    # Extract final model feature importance.
    extract_feature_importance(
        final_model,
        feature_names,
    )

    print("\n")
    print("=" * 65)
    print("VALIDATION AND FEATURE IMPORTANCE COMPLETED")
    print("=" * 65)

    print(f"\nSelected threshold: {threshold:.3f}")
    print(f"Reports directory: {REPORT_DIR}")
    print(f"Models directory: {MODEL_DIR}")


if __name__ == "__main__":
    main()
