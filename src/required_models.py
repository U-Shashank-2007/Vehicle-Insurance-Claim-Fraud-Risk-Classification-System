
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
SMOTE_NEIGHBORS = 5

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data" / "Processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data():

    print("\nLoading processed dataset...")

    X_train = load_npz(DATA_DIR / "X_train.npz")
    X_test = load_npz(DATA_DIR / "X_test.npz")

    # Reduce floating-point memory consumption.
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    groups_train = pd.read_csv(
        DATA_DIR / "groups_train.csv"
    ).iloc[:, 0].to_numpy()

    groups_test = pd.read_csv(
        DATA_DIR / "groups_test.csv"
    ).iloc[:, 0].to_numpy()

    # Verify customer separation.
    overlap = set(groups_train) & set(groups_test)

    if overlap:
        raise ValueError(
            f"Customer leakage detected: {len(overlap)} "
            "customers appear in both training and testing data."
        )

    print("\nDataset information")
    print("-" * 45)
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")
    print(f"Customer overlap : {len(overlap)}")

    print("\nTraining class distribution:")
    print(pd.Series(y_train).value_counts().sort_index())

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
    )


# ============================================================
# 2. EVALUATION
# ============================================================

def evaluate_model(name, model, X_test, y_test):

    print(f"\nEvaluating: {name}")
    print("=" * 60)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Default classification threshold.
    predictions = (probabilities >= 0.5).astype(int)

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
        "ROC_AUC": roc_auc_score(
            y_test, probabilities
        ),
        "PR_AUC": average_precision_score(
            y_test, probabilities
        ),
    }

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not Fraud", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nMetrics:")
    for metric, value in metrics.items():

        if metric != "Model":
            print(f"{metric:12s}: {value:.4f}")

    return metrics


# ============================================================
# 3. LOGISTIC REGRESSION + SMOTE
# ============================================================

def train_logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
):

    print("\n")
    print("=" * 60)
    print("LOGISTIC REGRESSION + SMOTE")
    print("=" * 60)

    pipeline = Pipeline([
        (
            "smote",
            SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=SMOTE_NEIGHBORS,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    print("\nTraining Logistic Regression...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    model_path = MODEL_DIR / "logistic_regression_smote.joblib"

    joblib.dump(pipeline, model_path)

    print(f"Model saved: {model_path}")

    metrics = evaluate_model(
        "Logistic Regression + SMOTE",
        pipeline,
        X_test,
        y_test,
    )

    return metrics


# ============================================================
# 4. RANDOM FOREST + GRID SEARCH + SMOTE
# ============================================================

def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    groups_train,
):

    print("\n")
    print("=" * 60)
    print("RANDOM FOREST GRID SEARCH + SMOTE")
    print("=" * 60)

    pipeline = Pipeline([
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

    # Two configurations, five folds = ten fits.
    param_grid = {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [20],
        "classifier__min_samples_leaf": [2],
        "classifier__max_features": ["sqrt"],
    }

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        verbose=2,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    print("\nStarting Grid Search...")
    print("Configurations: 2")
    print("Cross-validation folds: 5")
    print("Total model fits: 10")
    print("Parallel jobs: 1")

    grid_search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    best_model = grid_search.best_estimator_

    print("\nGrid Search completed.")

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest cross-validation ROC-AUC: "
        f"{grid_search.best_score_:.4f}"
    )

    # Save the optimized model.
    model_path = (
        MODEL_DIR / "random_forest_smote_gridsearch.joblib"
    )

    joblib.dump(best_model, model_path)

    print(f"\nModel saved: {model_path}")

    # Save all Grid Search results.
    cv_results = pd.DataFrame(grid_search.cv_results_)

    cv_results.to_csv(
        REPORT_DIR / "random_forest_gridsearch_results.csv",
        index=False,
    )

    # Save best parameters.
    with open(
        REPORT_DIR / "random_forest_best_params.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            grid_search.best_params_,
            file,
            indent=4,
        )

    metrics = evaluate_model(
        "Tuned Random Forest + SMOTE",
        best_model,
        X_test,
        y_test,
    )

    return metrics


# ============================================================
# 5. SAVE MODEL COMPARISON
# ============================================================

def save_comparison(results):

    comparison = pd.DataFrame(results)

    comparison_path = (
        REPORT_DIR / "required_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print("\n")
    print("=" * 75)
    print("FINAL MODEL COMPARISON")
    print("=" * 75)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nComparison saved: {comparison_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("VEHICLE INSURANCE FRAUD CLASSIFICATION")
    print("Required Model Experiments")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
    ) = load_data()

    results = []

    # Logistic Regression.
    logistic_metrics = train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    results.append(logistic_metrics)

    # Random Forest with Grid Search.
    rf_metrics = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        groups_train,
    )

    results.append(rf_metrics)

    # Save comparison.
    save_comparison(results)

    print("\n")
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated files:")
    print(f"Models  : {MODEL_DIR}")
    print(f"Reports : {REPORT_DIR}")


if __name__ == "__main__":
    main()
