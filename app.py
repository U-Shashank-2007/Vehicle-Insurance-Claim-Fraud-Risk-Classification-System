
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "final_random_forest_fraud_model.joblib"
PREPROCESSOR_PATH = ROOT / "Data" / "Processed" / "preprocessor.joblib"

THRESHOLD_PATH = ROOT / "reports" / "selected_threshold.json"
METRICS_PATH = ROOT / "reports" / "final_test_evaluation.csv"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"

st.set_page_config(
    page_title="ClaimGuard | Fraud Investigation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    [data-testid="stMetric"] {
        background-color: #151f30;
        border: 1px solid #263244;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #14243c, #172554);
        border: 1px solid #2c4267;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 36px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #a9bdd6;
        margin-bottom: 0;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 20px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .risk-high {
        color: #fca5a5;
        background: #451a1a;
        border: 1px solid #7f1d1d;
        padding: 15px;
        border-radius: 10px;
    }

    .risk-review {
        color: #fcd34d;
        background: #422006;
        border: 1px solid #92400e;
        padding: 15px;
        border-radius: 10px;
    }

    .risk-low {
        color: #86efac;
        background: #052e16;
        border: 1px solid #166534;
        padding: 15px;
        border-radius: 10px;
    }

    div.stButton > button[kind="primary"] {
        background: #2563eb;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 22px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
    }

    h1, h2, h3 {
        color: #f1f5f9;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #263244;
        border-radius: 10px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    required_files = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        THRESHOLD_PATH,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    threshold = float(threshold_data["threshold"])

    metrics = None
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)

    importance = None
    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH)

    return model, preprocessor, threshold, metrics, importance


try:
    model, preprocessor, THRESHOLD, metrics, importance = load_artifacts()
except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()


# ============================================================
# INPUT SCHEMA
# ============================================================

FEATURES = list(preprocessor.feature_names_in_)

NUMERICAL_FEATURES = list(
    preprocessor.transformers_[0][2]
)

CATEGORICAL_FEATURES = list(
    preprocessor.transformers_[1][2]
)

categorical_pipeline = (
    preprocessor.named_transformers_["categorical"]
)

encoder = categorical_pipeline.named_steps["encoder"]

CATEGORY_OPTIONS = {
    feature: list(categories)
    for feature, categories in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_,
    )
}


# ============================================================
# DEFAULT NUMERICAL VALUES
# ============================================================

NUMERIC_DEFAULTS = {
    "claim_amount": 5000.0,
    "deductible": 500.0,
    "age": 35,
    "annual_income": 60000.0,
    "credit_score": 700,
    "years_with_company": 5,
    "previous_claims": 0,
    "policy_duration_years": 1,
    "premium_amount": 1200.0,
    "coverage_amount": 50000.0,
    "witnesses": 0,
    "injuries": 0,
    "vehicle_year": 2022,
    "mileage": 30000,
    "vehicle_value": 25000.0,
    "claim_month": 1,
    "claim_day_of_week": 0,
    "reporting_delay_days": 1,
    "policy_tenure_years": 2.0,
    "vehicle_age_at_claim": 4,
}


NUMERIC_LIMITS = {
    "credit_score": (0.0, 1000.0),
    "claim_month": (1.0, 12.0),
    "claim_day_of_week": (0.0, 6.0),
    "vehicle_year": (1900.0, 2100.0),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_numeric_input(feature):

    default = NUMERIC_DEFAULTS.get(feature, 0.0)

    minimum, maximum = NUMERIC_LIMITS.get(
        feature,
        (0.0, None),
    )

    return st.number_input(
        feature.replace("_", " ").title(),
        min_value=minimum,
        max_value=maximum,
        value=float(default),
        step=1.0 if float(default).is_integer() else 100.0,
        key=f"input_{feature}",
        help=f"Model input: {feature}",
    )


def get_categorical_input(feature):

    options = CATEGORY_OPTIONS[feature]

    return st.selectbox(
        feature.replace("_", " ").title(),
        options=options,
        key=f"input_{feature}",
    )


def predict_claim(claim_df):

    # Preserve exact training input order.
    claim_df = claim_df[FEATURES]

    # Apply the saved preprocessing pipeline.
    transformed = preprocessor.transform(claim_df)

    # The model is a fitted SMOTE + Random Forest pipeline.
    score = float(
        model.predict_proba(transformed)[0, 1]
    )

    prediction = int(score >= THRESHOLD)

    return score, prediction


def risk_category(score):

    if score >= 0.60:
        return "High Priority", "high"

    if score >= THRESHOLD:
        return "Review Required", "review"

    return "Lower Priority", "low"


def display_prediction(score, prediction):

    category, level = risk_category(score)

    percentage = score * 100

    if level == "high":
        st.markdown(
            f"""
            <div class="risk-high">
                <h3>🔴 {category}</h3>
                <h1>{percentage:.1f}%</h1>
                <p>Model score exceeds the high-priority review level.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif level == "review":
        st.markdown(
            f"""
            <div class="risk-review">
                <h3>🟠 {category}</h3>
                <h1>{percentage:.1f}%</h1>
                <p>Claim meets the model's investigation threshold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f"""
            <div class="risk-low">
                <h3>🟢 {category}</h3>
                <h1>{percentage:.1f}%</h1>
                <p>Claim is below the current investigation threshold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(min(max(score, 0.0), 1.0))

    st.caption(
        f"Operating threshold: {THRESHOLD:.3f} "
        f"({THRESHOLD * 100:.1f}%)."
    )

    if prediction == 1:
        st.warning(
            "This claim has been flagged for human review. "
            "This is not a determination of fraud."
        )
    else:
        st.success(
            "This claim is below the current model threshold. "
            "Normal claim review procedures still apply."
        )


def validate_uploaded_data(data):

    missing = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            "CSV is missing required columns:\n"
            + ", ".join(missing)
        )

    return data[FEATURES].copy()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🛡️ ClaimGuard")
    st.caption("VEHICLE INSURANCE FRAUD SYSTEM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "New Claim",
            "Batch Assessment",
            "Model Insights",
        ],
    )

    st.divider()

    st.markdown("### Model status")
    st.success("Model loaded")

    st.markdown(
        f"""
        <p class="small-muted">
        Random Forest + SMOTE<br>
        Threshold: {THRESHOLD:.3f}<br>
        Input features: {len(FEATURES)}
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("ClaimGuard · Academic Project")


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">
        <h1>ClaimGuard</h1>
        <p>
            Vehicle Insurance Claim Fraud Risk Classification
            and Investigation Support System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## System Overview")

    st.write(
        "ClaimGuard uses a machine learning model to assess "
        "submitted vehicle insurance claims and help investigators "
        "prioritize claims for further review."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Dataset records", "100,000")
    c2.metric("Input features", len(FEATURES))
    c3.metric("Fraud threshold", f"{THRESHOLD:.3f}")
    c4.metric("Model", "Random Forest")

    st.markdown("---")
    st.markdown("## Final Model Performance")

    if metrics is not None and not metrics.empty:

        row = metrics.iloc[0]

        columns = st.columns(5)

        columns[0].metric(
            "Accuracy",
            f"{row['Accuracy']:.2%}",
        )

        columns[1].metric(
            "Precision",
            f"{row['Precision']:.2%}",
        )

        columns[2].metric(
            "Recall",
            f"{row['Recall']:.2%}",
        )

        columns[3].metric(
            "F1 Score",
            f"{row['F1']:.3f}",
        )

        columns[4].metric(
            "ROC-AUC",
            f"{row['ROC_AUC']:.3f}",
        )

    else:
        st.info("Model evaluation report not found.")

    st.markdown("---")

    st.markdown("## Workflow")

    workflow = st.columns(4)

    workflow[0].markdown(
        "**01 · Input**\n\nEnter claim information."
    )

    workflow[1].markdown(
        "**02 · Transform**\n\nApply saved preprocessing."
    )

    workflow[2].markdown(
        "**03 · Predict**\n\nCalculate model score."
    )

    workflow[3].markdown(
        "**04 · Review**\n\nPrioritize for investigation."
    )

    st.info(
        "ClaimGuard is an investigation-support tool. "
        "A model flag is not proof of fraud, and a low score "
        "does not guarantee that a claim is legitimate."
    )


# ============================================================
# PAGE 2: NEW CLAIM
# ============================================================

elif page == "New Claim":

    st.title("New Claim Assessment")

    st.write(
        "Enter the claim details below to calculate its "
        "fraud-risk score."
    )

    st.info(
        "Use information available at initial claim submission. "
        "Do not enter investigation outcomes or post-decision data."
    )

    with st.form("claim_form"):

        claim_values = {}

        sections = {
            "Claim Information": [
                "claim_amount",
                "deductible",
                "previous_claims",
                "witnesses",
                "injuries",
                "claim_month",
                "claim_day_of_week",
                "reporting_delay_days",
            ],

            "Customer Information": [
                "age",
                "gender",
                "marital_status",
                "occupation",
                "annual_income",
                "education_level",
                "credit_score",
                "years_with_company",
            ],

            "Policy Information": [
                "policy_type",
                "policy_duration_years",
                "premium_amount",
                "coverage_amount",
                "deductible_plan",
                "payment_method",
                "policy_active",
                "policy_tenure_years",
            ],

            "Incident Information": [
                "incident_type",
                "accident_severity",
                "incident_location",
                "weather_condition",
                "police_report",
                "hospital_visit",
                "property_damage",
                "claim_submitted_at_night",
                "multiple_claims_last_year",
                "delayed_reporting",
            ],

            "Vehicle Information": [
                "vehicle_make",
                "vehicle_model",
                "vehicle_year",
                "mileage",
                "vehicle_value",
                "ownership_type",
                "vehicle_age_at_claim",
            ],
        }

        for section, features in sections.items():

            st.markdown(
                f'<div class="section-title">{section}</div>',
                unsafe_allow_html=True,
            )

            columns = st.columns(2)

            for index, feature in enumerate(features):

                with columns[index % 2]:

                    if feature in NUMERICAL_FEATURES:
                        claim_values[feature] = (
                            get_numeric_input(feature)
                        )

                    else:
                        claim_values[feature] = (
                            get_categorical_input(feature)
                        )

        submitted = st.form_submit_button(
            "Assess Claim",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        claim_df = pd.DataFrame(
            [claim_values],
            columns=FEATURES,
        )

        try:

            score, prediction = predict_claim(claim_df)

            st.markdown("---")
            st.subheader("Assessment Result")

            display_prediction(score, prediction)

            result_df = claim_df.copy()

            result_df["fraud_score"] = score
            result_df["threshold"] = THRESHOLD
            result_df["flagged_for_review"] = prediction

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Assessment",
                data=csv,
                file_name="claim_assessment.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ============================================================
# PAGE 3: BATCH ASSESSMENT
# ============================================================

elif page == "Batch Assessment":

    st.title("Batch Claim Assessment")

    st.write(
        "Upload a CSV file containing multiple claims. "
        "ClaimGuard will calculate a score for each record."
    )

    st.info(
        "The uploaded CSV must contain all 41 original model input "
        "columns, using the same column names as the training data. "
        "It must include the engineered date features."
    )

    template = pd.DataFrame(columns=FEATURES)

    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="claimguard_input_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload claims CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            st.write(
                f"Uploaded records: **{len(uploaded_data):,}**"
            )

            st.dataframe(
                uploaded_data.head(),
                use_container_width=True,
            )

            if st.button(
                "Assess All Claims",
                type="primary",
            ):

                input_data = validate_uploaded_data(
                    uploaded_data
                )

                transformed = preprocessor.transform(
                    input_data
                )

                scores = model.predict_proba(
                    transformed
                )[:, 1]

                predictions = (
                    scores >= THRESHOLD
                ).astype(int)

                results = uploaded_data.copy()

                results["fraud_score"] = scores
                results["threshold"] = THRESHOLD

                results["risk_category"] = [
                    risk_category(score)[0]
                    for score in scores
                ]

                results["flagged_for_review"] = predictions

                results = results.sort_values(
                    "fraud_score",
                    ascending=False,
                )

                flagged_count = int(predictions.sum())

                st.session_state["batch_results"] = results

                st.success("Batch assessment completed.")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total claims",
                    f"{len(results):,}",
                )

                col2.metric(
                    "Flagged claims",
                    f"{flagged_count:,}",
                )

                col3.metric(
                    "Flagged share",
                    f"{flagged_count / len(results):.1%}"
                    if len(results) else "0%",
                )

        except Exception as error:
            st.error(f"Batch assessment failed: {error}")

    if "batch_results" in st.session_state:

        results = st.session_state["batch_results"]

        st.markdown("---")
        st.subheader("Ranked Claims")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Ranked Results",
            data=csv,
            file_name="claimguard_batch_results.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE 4: MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.title("Model Insights")

    st.write(
        "Explore the final model's evaluation metrics "
        "and feature importance."
    )

    st.markdown("## Evaluation Metrics")

    if metrics is not None and not metrics.empty:

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning("Evaluation metrics are unavailable.")

    st.markdown("---")
    st.markdown("## Feature Importance")

    if importance is not None and not importance.empty:

        top_n = st.slider(
            "Number of features to display",
            min_value=5,
            max_value=min(30, len(importance)),
            value=min(15, len(importance)),
        )

        top_features = importance.head(top_n).copy()

        top_features["Feature"] = (
            top_features["Feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        st.bar_chart(
            top_features.set_index("Feature")["Importance"],
            horizontal=True,
        )

        st.dataframe(
            top_features,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.warning("Feature importance report is unavailable.")

    st.markdown("---")

    st.markdown("## Model Configuration")

    col1, col2 = st.columns(2)

    col1.metric("Operating threshold", f"{THRESHOLD:.3f}")
    col2.metric("Input feature count", len(FEATURES))

    st.caption(
        "The threshold was selected using a separate validation "
        "partition by maximizing F1. The final test set was not "
        "used to select the threshold."
    )

    st.warning(
        "The displayed fraud score is the model's predicted score, "
        "not a calibrated real-world probability. Feature importance "
        "shows global model behavior, not proof of why an individual "
        "claim was flagged."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:10px;">
        ClaimGuard · Vehicle Insurance Fraud Risk Classification System
        <br>
        Academic Project · Machine Learning Investigation Support
    </div>
    """,
    unsafe_allow_html=True,
)
