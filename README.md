# Vehicle Insurance Claim Fraud Risk Classification System

A machine-learning decision-support dashboard for identifying vehicle
insurance claims that may warrant further fraud investigation. The
system compares Logistic Regression and a tuned Random Forest model,
then provides an interactive Streamlit interface for assessing
individual claims and ranking batches of claims by model score.

> **Important:** This project is an investigation-prioritization aid,
> not an automated fraud detector. A high score is not proof of fraud.
> All flagged claims require human review.

## Project overview

This project explores how supervised machine learning can help
investigators prioritize vehicle insurance claims for review. It uses
the Kaggle **Vehicle Insurance Claim Fraud Detection Dataset** and
includes exploratory data analysis, leakage-aware preprocessing,
class-imbalance handling, model comparison, hyperparameter tuning,
threshold selection, evaluation, and a Streamlit dashboard.

## Key features

-   Exploratory data analysis and feature engineering.
-   Leakage-aware feature selection.
-   Customer-group-aware train/test splitting.
-   Numerical imputation and categorical one-hot encoding.
-   SMOTE applied within model-training pipelines.
-   Logistic Regression baseline and Grid-Search-tuned Random Forest.
-   Evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC.
-   Validation-based decision-threshold selection.
-   Feature-importance analysis.
-   Streamlit dashboard for single-claim and batch assessment.

## Repository structure

``` text
CA-1/
├── Data/
│   ├── Raw/
│   │   └── insurance_claim_fraud_dataset.csv
│   └── Processed/
│       ├── feature_names.json
│       ├── groups_test.csv
│       ├── groups_train.csv
│       ├── preprocessor.joblib
│       ├── X_test.npz
│       ├── X_train.npz
│       ├── y_test.npy
│       └── y_train.npy
├── models/
├── reports/
├── src/
│   ├── models/
│   ├── reports/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   ├── model.py
│   └── train.py
├── app.py
├── requirements.txt
└── README.md
```

Generated data, model, and report artifacts are excluded from version
control through `.gitignore`. The dashboard therefore requires the
corresponding locally generated artifacts to be present before it can
run.

## Dataset

**Source:** Kaggle --- Vehicle Insurance Claim Fraud Detection Dataset.

The working dataset contains 100,000 records and 62 columns. The target
column is `fraudulent`:

  Target     Records
  -------- ---------
  No          79,958
  Yes         20,042

Approximately 20% of the records are labelled fraudulent.

To reduce target leakage, `fraud_risk_score` was excluded because it
perfectly separated the target in this dataset. Post-decision fields
such as claim status and payout-related amounts were also excluded.
Identifiers and personal names were excluded from model inputs.
`customer_id` is retained only to support group-aware splitting.

## Methodology

### Preprocessing and feature engineering

Date fields are used to derive `claim_month`, `claim_day_of_week`,
`reporting_delay_days`, `policy_tenure_years`, and
`vehicle_age_at_claim`. Invalid or impossible date-derived values are
treated as missing.

Numerical columns are imputed using the training-set median. Categorical
columns are imputed using the training-set most frequent value and
one-hot encoded. Preprocessing transformations are fitted on training
data only.

### Train/test split

A `StratifiedGroupKFold` split is used with `customer_id` as the
grouping variable. The first fold is held out for testing:

-   Training set: 80,000 records.
-   Test set: 20,000 records.
-   No customer overlap between training and test sets.

### Class imbalance

SMOTE is applied within the model-training pipeline, so resampling
occurs during fitting rather than before the train/test split.

### Models and tuning

The required model comparison includes Logistic Regression with SMOTE
and a Random Forest with SMOTE and Grid Search tuning.

Selected Random Forest parameters:

``` python
{
    "classifier__max_depth": 20,
    "classifier__max_features": "sqrt",
    "classifier__min_samples_leaf": 2,
    "classifier__n_estimators": 250
}
```

### Decision threshold

The final Random Forest decision threshold was selected using a separate
validation split from the training data, maximizing validation F1-score.
The selected threshold was approximately **0.23**. The held-out test set
was not used to select this threshold.

## Results

### Model comparison at the default 0.50 threshold

  ---------------------------------------------------------------------------------
  Model            Accuracy   Precision     Recall   F1-score    ROC-AUC     PR-AUC
  -------------- ---------- ----------- ---------- ---------- ---------- ----------
  Logistic           0.6454      0.2476     0.3772     0.2990     0.5597     0.2503
  Regression +                                                           
  SMOTE                                                                  

  Tuned Random       0.8030      0.6736     0.0324     0.0619     0.6482     0.3617
  Forest + SMOTE                                                         
  ---------------------------------------------------------------------------------

### Tuned Random Forest at the selected validation threshold

  ------------------------------------------------------------------------------
    Threshold   Accuracy   Precision     Recall   F1-score    ROC-AUC     PR-AUC
  ----------- ---------- ----------- ---------- ---------- ---------- ----------
       \~0.23     0.6947      0.3364     0.5384     0.4141     0.6482     0.3617

  ------------------------------------------------------------------------------

The lower operating threshold substantially increases recall compared
with using 0.50, at the cost of lower precision. This trade-off is
intended for an investigator-facing workflow: potentially suspicious
claims are prioritized for human review rather than automatically
rejected.

At the selected threshold, the model identifies approximately 53.8% of
fraudulent claims in the held-out test set. Approximately 33.6% of the
claims it flags are fraudulent. The score is not a calibrated
probability and should not be interpreted as a final fraud decision.

## Streamlit dashboard

The dashboard contains:

-   **Overview:** project summary and evaluation metrics.
-   **New Claim:** enter claim details and obtain a model score and risk
    flag.
-   **Batch Assessment:** upload a CSV, score and rank claims, and
    download the results.
-   **Model Insights:** inspect evaluation metrics and
    feature-importance information.

### Run locally

1.  Clone the repository:

    ``` powershell
    git clone https://github.com/U-Shashank-2007/Vehicle-Insurance-Claim-Fraud-Risk-Classification-System.git
    cd Vehicle-Insurance-Claim-Fraud-Risk-Classification-System
    ```

2.  Create and activate a virtual environment:

    ``` powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3.  Install dependencies:

    ``` powershell
    python -m pip install -r requirements.txt
    ```

4.  Download the dataset from Kaggle and place it at:

    ``` text
    Data/Raw/insurance_claim_fraud_dataset.csv
    ```

5.  Ensure the locally generated preprocessing, model, threshold, and
    report artifacts are available at the paths expected by `app.py`.
    These artifacts are excluded from the repository and must be
    generated by the project workflow or supplied separately.

6.  Run the dashboard from the project root:

    ``` powershell
    streamlit run app.py
    ```

    Or:

    ``` powershell
    python -m streamlit run app.py
    ```

## Batch CSV input

The batch assessment page expects a CSV containing the model input
features with the column names and schema used by the preprocessing
pipeline. Include the engineered fields required by the model:

-   `claim_month`
-   `claim_day_of_week`
-   `reporting_delay_days`
-   `policy_tenure_years`
-   `vehicle_age_at_claim`

Do not provide target labels or post-investigation/post-decision
information as model inputs. The saved preprocessor handles missing
values and categorical encoding.

## Limitations and responsible use

-   The dataset's labels and feature distributions may not represent
    real insurer claim populations.
-   Model performance is moderate and does not justify automatic claim
    rejection or accusations of fraud.
-   The selected threshold reflects one validation split and may not
    transfer to other datasets or business settings.
-   SMOTE is currently applied after one-hot encoding. This can create
    fractional values in encoded categorical columns and is a
    methodological limitation.
-   Impurity-based Random Forest feature importance can be biased and is
    not causal evidence.
-   The displayed model score is not established as a calibrated
    probability.
-   Fairness, privacy, data quality, drift monitoring, and external
    validation require further assessment before real-world use.

The system should support trained investigators only. Human review and
independent evidence are required for every claim decision.

## Future work

-   Add claim-level explanations for individual assessments.
-   Evaluate probability calibration and threshold stability.
-   Investigate imbalance-handling approaches that preserve categorical
    feature structure.
-   Validate on independent, real-world data where permitted.
-   Add investigator feedback and audit history.
-   Assess fairness and model drift before operational deployment.

## Team

Developed as a three-student academic project.

## License

Add the license selected by the project team before distributing or
reusing this repository.
