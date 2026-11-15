"""
ClaimGuard - model service layer.

This module is the ONLY place that knows about model files, preprocessing,
thresholds and feature names. The desktop UI talks to it through a small,
plain API (dataclasses and DataFrames) and never touches the model directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths (adjust here if your project layout differs)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "final_random_forest_fraud_model.joblib"
PREPROCESSOR_PATH = ROOT / "Data" / "Processed" / "preprocessor.joblib"
THRESHOLD_PATH = ROOT / "reports" / "selected_threshold.json"
METRICS_PATH = ROOT / "reports" / "final_test_evaluation.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"

# Optional labelled hold-out file used for the one-click classification report.
# (Features + a fraud label column.) If it does not exist, the UI asks the
# user to choose a labelled file instead.
DEFAULT_EVALUATION_PATH = ROOT / "Data" / "Processed" / "test_claims.csv"

LABEL_CANDIDATES = [
    "fraud", "is_fraud", "fraud_reported", "fraudulent",
    "fraud_label", "label", "target", "class",
]

HIGH_PRIORITY_SCORE = 0.60

# ---------------------------------------------------------------------------
# Form definition (presentation grouping + friendly labels)
# ---------------------------------------------------------------------------
SECTIONS = {
    "Claim": [
        "claim_amount", "deductible", "previous_claims", "witnesses",
        "injuries", "claim_month", "claim_day_of_week",
        "reporting_delay_days",
    ],
    "Customer": [
        "age", "gender", "marital_status", "occupation", "annual_income",
        "education_level", "credit_score", "years_with_company",
    ],
    "Policy": [
        "policy_type", "policy_duration_years", "premium_amount",
        "coverage_amount", "deductible_plan", "payment_method",
        "policy_active", "policy_tenure_years",
    ],
    "Incident": [
        "incident_type", "accident_severity", "incident_location",
        "weather_condition", "police_report", "hospital_visit",
        "property_damage", "claim_submitted_at_night",
        "multiple_claims_last_year", "delayed_reporting",
    ],
    "Vehicle": [
        "vehicle_make", "vehicle_model", "vehicle_year", "mileage",
        "vehicle_value", "ownership_type", "vehicle_age_at_claim",
    ],
}

LABELS = {
    "claim_day_of_week": "Day of week of claim (0 = Monday)",
    "claim_month": "Month of claim (1-12)",
    "reporting_delay_days": "Days taken to report",
    "years_with_company": "Years as a customer",
    "policy_tenure_years": "Policy tenure (years)",
    "claim_submitted_at_night": "Submitted at night",
    "multiple_claims_last_year": "Multiple claims last year",
    "vehicle_age_at_claim": "Vehicle age at claim (years)",
    "deductible_plan": "Deductible plan",
}

NUMERIC_DEFAULTS = {
    "claim_amount": 5000.0, "deductible": 500.0, "age": 35,
    "annual_income": 60000.0, "credit_score": 700, "years_with_company": 5,
    "previous_claims": 0, "policy_duration_years": 1,
    "premium_amount": 1200.0, "coverage_amount": 50000.0, "witnesses": 0,
    "injuries": 0, "vehicle_year": 2022, "mileage": 30000,
    "vehicle_value": 25000.0, "claim_month": 1, "claim_day_of_week": 0,
    "reporting_delay_days": 1, "policy_tenure_years": 2.0,
    "vehicle_age_at_claim": 4,
}

NUMERIC_LIMITS = {
    "credit_score": (0.0, 1000.0),
    "claim_month": (1.0, 12.0),
    "claim_day_of_week": (0.0, 6.0),
    "vehicle_year": (1900.0, 2100.0),
}


MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
        "Sunday"]


def _range(lo, hi, unit=None, descending=False):
    vals = list(range(lo, hi + 1))
    if descending:
        vals.reverse()
    out = []
    for v in vals:
        if unit:
            out.append((f"{v} {unit}" + ("" if v == 1 else "s"), v))
        else:
            out.append((str(v), v))
    return out


def _delay_options():
    out = []
    for v in [0, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 45, 60, 90]:
        out.append(("Same day" if v == 0 else f"{v} day" + ("" if v == 1 else "s"), v))
    return out


# Small, bounded numeric inputs are shown as dropdowns instead of free typing.
CHOICE_NUMBERS = {
    "claim_month": [(m, i) for i, m in enumerate(MONTHS, 1)],
    "claim_day_of_week": [(d, i) for i, d in enumerate(DAYS)],
    "previous_claims": _range(0, 10),
    "witnesses": _range(0, 6),
    "injuries": _range(0, 6),
    "reporting_delay_days": _delay_options(),
    "years_with_company": _range(0, 30, "year"),
    "policy_duration_years": _range(1, 10, "year"),
    "vehicle_age_at_claim": _range(0, 25, "year"),
    "vehicle_year": _range(1990, date.today().year + 1, descending=True),
}

MONEY_FIELDS = {
    "claim_amount", "deductible", "annual_income", "premium_amount",
    "coverage_amount", "vehicle_value",
}


def _display(option) -> str:
    text = str(option).replace("_", " ")
    return text[:1].upper() + text[1:]


def _pretty(name: str) -> str:
    return LABELS.get(name, name.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------
@dataclass
class FieldSpec:
    name: str                     # internal key (used only to round-trip values)
    label: str                    # what the user sees
    kind: str                     # "number" | "choice"
    default: object = None
    minimum: float = 0.0
    maximum: Optional[float] = None
    is_integer: bool = False
    decimals: int = 0             # for "number" fields: digits after the point
    options: list = field(default_factory=list)         # real values
    option_labels: list = field(default_factory=list)   # what the user sees


@dataclass
class Assessment:
    score: float                  # 0..1
    flagged: bool
    level: str                    # "high" | "review" | "low"
    title: str                    # e.g. "High priority"
    message: str                  # user-facing explanation


@dataclass
class ClassReportRow:
    name: str
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    support: int


@dataclass
class ClassificationSummary:
    rows: list                    # list[ClassReportRow]
    accuracy: float
    precision: float              # for the fraud class
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion: list               # [[TN, FP], [FN, TP]]
    total: int
    fraud_total: int


class ServiceError(Exception):
    """Raised with a user-presentable message."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ModelService:
    def __init__(self) -> None:
        required = [MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Required files are missing:\n" + "\n".join(missing)
            )

        self._model = joblib.load(MODEL_PATH)
        self._pre = joblib.load(PREPROCESSOR_PATH)

        with open(THRESHOLD_PATH, "r", encoding="utf-8") as fh:
            self._threshold = float(json.load(fh)["threshold"])

        self._features = list(self._pre.feature_names_in_)
        self._numeric = set(self._pre.transformers_[0][2])
        categorical = list(self._pre.transformers_[1][2])
        encoder = self._pre.named_transformers_["categorical"].named_steps["encoder"]
        self._choices = {
            f: list(c) for f, c in zip(categorical, encoder.categories_)
        }

    # ------------------------------------------------------------------ form
    def form_sections(self) -> list[tuple[str, list[FieldSpec]]]:
        """Grouped input fields for the claim form."""
        placed: set[str] = set()
        result: list[tuple[str, list[FieldSpec]]] = []

        for title, names in SECTIONS.items():
            specs = [self._spec(n) for n in names if n in self._features]
            placed.update(s.name for s in specs)
            if specs:
                result.append((title, specs))

        leftovers = [self._spec(n) for n in self._features if n not in placed]
        if leftovers:
            result.append(("Additional details", leftovers))
        return result

    def _spec(self, name: str) -> FieldSpec:
        if name in self._numeric:
            default = NUMERIC_DEFAULTS.get(name, 0.0)
            if name in CHOICE_NUMBERS:
                pairs = CHOICE_NUMBERS[name]
                values = [v for _, v in pairs]
                chosen = default if default in values else values[0]
                return FieldSpec(
                    name=name, label=_pretty(name), kind="choice",
                    default=chosen, options=values,
                    option_labels=[l for l, _ in pairs],
                )
            lo, hi = NUMERIC_LIMITS.get(name, (0.0, None))
            if name in MONEY_FIELDS:
                decimals = 2
            else:
                decimals = 0 if float(default).is_integer() else 2
            return FieldSpec(
                name=name, label=_pretty(name), kind="number",
                default=float(default), minimum=lo, maximum=hi,
                is_integer=decimals == 0, decimals=decimals,
            )
        options = self._choices.get(name, [])
        return FieldSpec(
            name=name, label=_pretty(name), kind="choice",
            default=options[0] if options else "", options=options,
            option_labels=[_display(o) for o in options],
        )

    # ------------------------------------------------------------ assessment
    def _level(self, score: float) -> tuple[str, str, str]:
        if score >= HIGH_PRIORITY_SCORE:
            return ("high", "High priority",
                    "This claim shows a strong risk pattern and should be "
                    "reviewed by an investigator first.")
        if score >= self._threshold:
            return ("review", "Review recommended",
                    "This claim shows enough risk signals to be worth a "
                    "closer look.")
        return ("low", "Lower priority",
                "This claim shows few risk signals. Normal review "
                "procedures still apply.")

    def _scores(self, frame: pd.DataFrame) -> np.ndarray:
        data = frame[self._features]
        return self._model.predict_proba(self._pre.transform(data))[:, 1]

    def assess(self, values: dict) -> Assessment:
        frame = pd.DataFrame([values], columns=self._features)
        score = float(self._scores(frame)[0])
        level, title, message = self._level(score)
        return Assessment(
            score=score, flagged=score >= self._threshold,
            level=level, title=title, message=message,
        )

    def export_frame(self, values: dict, result: Assessment) -> pd.DataFrame:
        out = pd.DataFrame([values], columns=self._features)
        out.insert(0, "Flagged for review", "Yes" if result.flagged else "No")
        out.insert(0, "Priority", result.title)
        out.insert(0, "Risk score (%)", round(result.score * 100, 1))
        return out

    # ----------------------------------------------------------------- batch
    def template_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self._features)

    def _check_columns(self, data: pd.DataFrame) -> None:
        missing = [f for f in self._features if f not in data.columns]
        if missing:
            shown = ", ".join(missing[:12])
            more = f" and {len(missing) - 12} more" if len(missing) > 12 else ""
            raise ServiceError(
                f"The file is missing {len(missing)} required column(s): "
                f"{shown}{more}.\nDownload the template to see the expected layout."
            )

    def assess_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(data)
        scores = self._scores(data)
        levels = [self._level(s)[1] for s in scores]

        out = data.copy()
        out.insert(0, "Flagged for review",
                   np.where(scores >= self._threshold, "Yes", "No"))
        out.insert(0, "Priority", levels)
        out.insert(0, "Risk score (%)", np.round(scores * 100, 1))
        return out.sort_values("Risk score (%)", ascending=False).reset_index(drop=True)

    # ------------------------------------------------- classification report
    def has_default_evaluation_data(self) -> bool:
        return DEFAULT_EVALUATION_PATH.exists()

    def load_default_evaluation_data(self) -> pd.DataFrame:
        return pd.read_csv(DEFAULT_EVALUATION_PATH)

    def detect_label_column(self, data: pd.DataFrame) -> Optional[str]:
        lowered = {c.lower(): c for c in data.columns}
        for cand in LABEL_CANDIDATES:
            if cand in lowered:
                return lowered[cand]
        return None

    def label_column_choices(self, data: pd.DataFrame) -> list[str]:
        return [c for c in data.columns if c not in self._features]

    @staticmethod
    def _to_binary(series: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(series):
            return (series.astype(float) > 0).astype(int).to_numpy()
        positives = {"1", "y", "yes", "true", "fraud", "fraudulent", "t"}
        return series.astype(str).str.strip().str.lower().isin(positives).astype(int).to_numpy()

    def classification_report(
        self, data: pd.DataFrame, label_column: Optional[str] = None
    ) -> ClassificationSummary:
        """Ask the model to score labelled claims and return the full report."""
        label_column = label_column or self.detect_label_column(data)
        if label_column is None or label_column not in data.columns:
            raise ServiceError("The file needs a column with the true fraud label.")
        self._check_columns(data)

        y_true = self._to_binary(data[label_column])
        if len(np.unique(y_true)) < 2:
            raise ServiceError(
                "The label column contains only one class, so a report "
                "cannot be produced."
            )

        scores = self._scores(data)
        y_pred = (scores >= self._threshold).astype(int)

        raw = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            output_dict=True, zero_division=0,
        )

        rows = []
        for key, name in [("Legitimate", "Legitimate"), ("Fraud", "Fraud")]:
            r = raw[key]
            rows.append(ClassReportRow(name, r["precision"], r["recall"],
                                       r["f1-score"], int(r["support"])))
        rows.append(ClassReportRow("Accuracy", None, None, raw["accuracy"], len(y_true)))
        for key, name in [("macro avg", "Macro average"),
                          ("weighted avg", "Weighted average")]:
            r = raw[key]
            rows.append(ClassReportRow(name, r["precision"], r["recall"],
                                       r["f1-score"], int(r["support"])))

        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            auc = None

        return ClassificationSummary(
            rows=rows,
            accuracy=float(raw["accuracy"]),
            precision=float(raw["Fraud"]["precision"]),
            recall=float(raw["Fraud"]["recall"]),
            f1=float(raw["Fraud"]["f1-score"]),
            roc_auc=auc,
            confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            total=int(len(y_true)),
            fraud_total=int(y_true.sum()),
        )

    # -------------------------------------------------------- saved insights
    def headline_metrics(self) -> Optional[dict]:
        """Final test-set metrics saved at training time (for the home page)."""
        if not METRICS_PATH.exists():
            return None
        row = pd.read_csv(METRICS_PATH).iloc[0]
        try:
            return {
                "Accuracy": float(row["Accuracy"]),
                "Precision": float(row["Precision"]),
                "Recall": float(row["Recall"]),
                "F1 score": float(row["F1"]),
                "ROC-AUC": float(row["ROC_AUC"]),
            }
        except KeyError:
            return None

    def top_factors(self, n: int = 10) -> list[tuple[str, float]]:
        """Most influential claim attributes overall, scaled 0..1."""
        if not IMPORTANCE_PATH.exists():
            return []
        df = pd.read_csv(IMPORTANCE_PATH).head(n).copy()
        names = (
            df["Feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
            .map(_pretty)
        )
        values = df["Importance"].astype(float)
        top = values.max() or 1.0
        return list(zip(names, (values / top).tolist()))