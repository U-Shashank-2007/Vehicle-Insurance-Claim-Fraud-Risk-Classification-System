"""
Model definitions for the Insurance Fraud Detection project.

The models consume the transformed sparse matrices saved by preprocessing.ipynb.
All preprocessing (imputation and one-hot encoding) is handled upstream.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier


def create_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create the Random Forest baseline."""
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def create_xgboost(
    scale_pos_weight: float,
    random_state: int = 42,
) -> XGBClassifier:
    """Create the XGBoost baseline."""
    return XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=random_state,
    )


def create_voting_ensemble(
    random_forest: RandomForestClassifier,
    xgboost: XGBClassifier,
) -> VotingClassifier:
    """Create a soft-voting ensemble from RF and XGBoost."""
    return VotingClassifier(
        estimators=[
            ("random_forest", random_forest),
            ("xgboost", xgboost),
        ],
        voting="soft",
        weights=[1, 1],
        n_jobs=1,
    )


class GroupAwareStackingClassifier(ClassifierMixin, BaseEstimator):
    """
    Stacking classifier that uses StratifiedGroupKFold for out-of-fold
    predictions, preventing the same customer from appearing in both
    the fold-training and fold-validation portions.

    The meta-model is Logistic Regression trained on out-of-fold
    probability predictions from Random Forest and XGBoost.
    """

    def __init__(
        self,
        random_forest,
        xgboost,
        n_splits: int = 5,
        random_state: int = 42,
        meta_classifier=None,
    ):
        self.random_forest = random_forest
        self.xgboost = xgboost
        self.n_splits = n_splits
        self.random_state = random_state
        self.meta_classifier = meta_classifier

    def fit(self, X, y, groups):
        if groups is None:
            raise ValueError(
                "groups must be provided for group-aware stacking."
            )

        y_array = y.to_numpy() if hasattr(y, "to_numpy") else y
        groups_array = (
            groups.to_numpy() if hasattr(groups, "to_numpy") else groups
        )

        import numpy as np

        self.classes_ = np.asarray(sorted(set(y_array)))
        if len(self.classes_) != 2:
            raise ValueError("This implementation supports binary classification.")

        splitter = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        # Two columns: RF positive-class probability, XGBoost positive-class probability
        oof_predictions = None
        oof_counts = [0] * len(y_array)

        for fold, (fit_idx, valid_idx) in enumerate(
            splitter.split(X, y_array, groups_array),
            start=1,
        ):
            rf_fold = clone(self.random_forest)
            xgb_fold = clone(self.xgboost)

            X_fold_train = X[fit_idx]
            X_fold_valid = X[valid_idx]
            y_fold_train = y_array[fit_idx]

            rf_fold.fit(X_fold_train, y_fold_train)
            xgb_fold.fit(X_fold_train, y_fold_train)

            rf_probability = rf_fold.predict_proba(X_fold_valid)[:, 1]
            xgb_probability = xgb_fold.predict_proba(X_fold_valid)[:, 1]

            fold_predictions = np.column_stack(
                [rf_probability, xgb_probability]
            )

            if oof_predictions is None:
                import numpy as np
                oof_predictions = np.zeros(
                    (len(y_array), fold_predictions.shape[1]),
                    dtype=float,
                )

            oof_predictions[valid_idx] = fold_predictions
            for index in valid_idx:
                oof_counts[index] += 1

            print(f"  Stacking OOF fold {fold}/{self.n_splits} complete.")

        if not all(count == 1 for count in oof_counts):
            raise RuntimeError(
                "Every training row must receive exactly one out-of-fold prediction."
            )

        # Fit the final base estimators on the complete training partition.
        self.random_forest_ = clone(self.random_forest).fit(X, y_array)
        self.xgboost_ = clone(self.xgboost).fit(X, y_array)

        if self.meta_classifier is None:
            self.meta_classifier_ = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=self.random_state,
            )
        else:
            self.meta_classifier_ = clone(self.meta_classifier)

        self.meta_classifier_.fit(oof_predictions, y_array)
        return self

    def predict_proba(self, X):
        import numpy as np

        rf_probability = self.random_forest_.predict_proba(X)[:, 1]
        xgb_probability = self.xgboost_.predict_proba(X)[:, 1]

        meta_features = np.column_stack(
            [rf_probability, xgb_probability]
        )
        return self.meta_classifier_.predict_proba(meta_features)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
