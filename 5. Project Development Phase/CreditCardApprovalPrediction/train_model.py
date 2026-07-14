"""
Credit Card Approval Prediction System
Complete model-training file

Dataset files:
    dataset/application_record.csv
    dataset/credit_record.csv

Outputs:
    model.pkl
    model_metadata.pkl
    model_comparison.csv
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

APPLICATION_FILE = DATASET_DIR / "application_record.csv"
CREDIT_FILE = DATASET_DIR / "credit_record.csv"

MODEL_FILE = BASE_DIR / "model.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"
COMPARISON_FILE = BASE_DIR / "model_comparison.csv"


# ============================================================
# 2. LOAD DATASETS
# ============================================================

def load_datasets():
    """Load application and credit-record datasets."""

    if not APPLICATION_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {APPLICATION_FILE}\n"
            "Place application_record.csv inside the dataset folder."
        )

    if not CREDIT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {CREDIT_FILE}\n"
            "Place credit_record.csv inside the dataset folder."
        )

    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    application = pd.read_csv(APPLICATION_FILE)
    credit = pd.read_csv(CREDIT_FILE, dtype={"STATUS": str})

    print("Application Dataset Shape:", application.shape)
    print("Credit Dataset Shape     :", credit.shape)

    print("\nApplication Dataset First Five Rows:")
    print(application.head())

    print("\nCredit Dataset First Five Rows:")
    print(credit.head())

    return application, credit


# ============================================================
# 3. DATA UNDERSTANDING
# ============================================================

def display_dataset_information(application, credit):
    """Display missing values and duplicate-record information."""

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("\nMissing values in application dataset:")
    print(application.isnull().sum())

    print("\nMissing values in credit dataset:")
    print(credit.isnull().sum())

    print("\nDuplicate rows in application dataset:",
          application.duplicated().sum())

    print("Duplicate rows in credit dataset:",
          credit.duplicated().sum())


# ============================================================
# 4. CREATE TARGET VARIABLE
# ============================================================

def create_credit_target(credit):
    """
    Create one approval result for each applicant.

    STATUS meaning:
        X = No loan for the month
        C = Loan paid
        0 = 1–29 days overdue
        1 = 30–59 days overdue
        2 = 60–89 days overdue
        3 = 90–119 days overdue
        4 = 120–149 days overdue
        5 = 150 or more days overdue

    Project rule:
        Approved = 1 for X, C and 0
        Rejected = 0 for 1, 2, 3, 4 and 5

    If an applicant has at least one bad monthly record,
    the applicant is assigned Rejected.
    """

    credit = credit.copy()

    credit["STATUS"] = credit["STATUS"].astype(str).str.strip()

    good_statuses = ["X", "C", "0"]

    credit["Approved"] = np.where(
        credit["STATUS"].isin(good_statuses),
        1,
        0
    )

    # One target result for each applicant.
    # min() returns 0 if even one bad record exists.
    credit_summary = (
        credit.groupby("ID", as_index=False)["Approved"]
        .min()
    )

    print("\nCredit target created successfully.")
    print("Number of unique applicants in credit records:",
          credit_summary.shape[0])

    print("\nTarget distribution before merging:")
    print(credit_summary["Approved"].value_counts())

    return credit_summary


# ============================================================
# 5. MERGE AND CLEAN DATA
# ============================================================

def prepare_dataset(application, credit_summary):
    """Merge datasets and perform feature engineering."""

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING AND FEATURE ENGINEERING")
    print("=" * 70)

    application = application.copy()

    # Remove duplicate application records.
    application = application.drop_duplicates(subset=["ID"])

    # Merge one application row with one target result.
    data = pd.merge(
        application,
        credit_summary,
        on="ID",
        how="inner"
    )

    print("Merged Dataset Shape:", data.shape)

    # ID is only an identifier and should not be used for prediction.
    data = data.drop(columns=["ID"])

    # Convert negative birth days into age in years.
    data["AGE"] = (
        -data["DAYS_BIRTH"] / 365.25
    ).round().astype(int)

    data = data.drop(columns=["DAYS_BIRTH"])

    # DAYS_EMPLOYED = 365243 is a special value for applicants
    # who are unemployed, pensioners or not currently working.
    data["EMPLOYMENT_YEARS"] = np.where(
        data["DAYS_EMPLOYED"] == 365243,
        0,
        -data["DAYS_EMPLOYED"] / 365.25
    )

    data["EMPLOYMENT_YEARS"] = (
        data["EMPLOYMENT_YEARS"]
        .clip(lower=0)
        .round(1)
    )

    data = data.drop(columns=["DAYS_EMPLOYED"])

    # Fill missing occupation values.
    data["OCCUPATION_TYPE"] = (
        data["OCCUPATION_TYPE"]
        .fillna("Unknown")
        .astype(str)
    )

    # Remove remaining duplicate feature rows.
    # Target is included so conflicting labels are not combined accidentally.
    data = data.drop_duplicates()

    # Replace infinite values with missing values.
    data = data.replace([np.inf, -np.inf], np.nan)

    # Fill missing numerical values using median.
    numerical_columns = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in numerical_columns:
        if column != "Approved":
            data[column] = data[column].fillna(
                data[column].median()
            )

    # Fill missing categorical values.
    categorical_columns = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
        )

    print("Prepared Dataset Shape:", data.shape)

    print("\nFinal target distribution:")
    print(data["Approved"].value_counts())

    print("\nFinal target percentages:")
    print(
        data["Approved"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nFinal dataset first five rows:")
    print(data.head())

    return data


# ============================================================
# 6. CREATE PREPROCESSOR
# ============================================================

def create_preprocessor(X):
    """
    Create preprocessing transformer.

    Numerical columns:
        Missing-value handling is already completed.
        StandardScaler scales the values.

    Categorical columns:
        OneHotEncoder converts categories into numerical columns.
        handle_unknown='ignore' prevents errors for unseen values.
    """

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("\nCategorical Features:")
    for feature in categorical_features:
        print(" -", feature)

    print("\nNumerical Features:")
    for feature in numerical_features:
        print(" -", feature)

    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor, categorical_features, numerical_features


# ============================================================
# 7. CREATE MODELS
# ============================================================

def create_models():
    """Create classification models."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    # XGBoost is added when it is installed.
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )

    except ImportError:
        print(
            "\nXGBoost is not installed. "
            "The remaining three models will still be trained."
        )
        print("Install it using: pip install xgboost")

    return models


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

def evaluate_prediction(y_test, prediction):
    """Calculate model evaluation metrics."""

    return {
        "Accuracy": accuracy_score(y_test, prediction),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            prediction
        ),
        "Macro F1 Score": f1_score(
            y_test,
            prediction,
            average="macro",
            zero_division=0
        ),
        "Rejected Precision": precision_score(
            y_test,
            prediction,
            pos_label=0,
            zero_division=0
        ),
        "Rejected Recall": recall_score(
            y_test,
            prediction,
            pos_label=0,
            zero_division=0
        ),
        "Rejected F1 Score": f1_score(
            y_test,
            prediction,
            pos_label=0,
            zero_division=0
        )
    }


def train_and_compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor
):
    """Train models, evaluate them and select the best pipeline."""

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 70)

    models = create_models()

    results = []
    trained_pipelines = {}

    for model_name, classifier in models.items():
        print("\n" + "-" * 70)
        print("Training:", model_name)
        print("-" * 70)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        try:
            pipeline.fit(X_train, y_train)

            prediction = pipeline.predict(X_test)

            metrics = evaluate_prediction(
                y_test,
                prediction
            )

            metrics["Model"] = model_name
            results.append(metrics)

            trained_pipelines[model_name] = pipeline

            print(f"Accuracy          : {metrics['Accuracy']:.4f}")
            print(
                "Balanced Accuracy : "
                f"{metrics['Balanced Accuracy']:.4f}"
            )
            print(
                "Macro F1 Score     : "
                f"{metrics['Macro F1 Score']:.4f}"
            )
            print(
                "Rejected Precision : "
                f"{metrics['Rejected Precision']:.4f}"
            )
            print(
                "Rejected Recall    : "
                f"{metrics['Rejected Recall']:.4f}"
            )
            print(
                "Rejected F1 Score  : "
                f"{metrics['Rejected F1 Score']:.4f}"
            )

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, prediction))

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    prediction,
                    target_names=[
                        "Rejected",
                        "Approved"
                    ],
                    zero_division=0
                )
            )

        except Exception as error:
            print(
                f"{model_name} could not be trained: {error}"
            )

    if not results:
        raise RuntimeError(
            "No machine-learning model was trained successfully."
        )

    comparison = pd.DataFrame(results)

    comparison = comparison[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1 Score",
            "Rejected Precision",
            "Rejected Recall",
            "Rejected F1 Score"
        ]
    ]

    comparison = comparison.sort_values(
        by=[
            "Macro F1 Score",
            "Balanced Accuracy",
            "Rejected F1 Score"
        ],
        ascending=False
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(comparison.to_string(index=False))

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    # Macro F1 is used because this dataset can be highly imbalanced.
    best_model_name = comparison.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\nBest Model:", best_model_name)

    return best_pipeline, best_model_name, comparison


# ============================================================
# 9. SAVE MODEL AND METADATA
# ============================================================

def save_outputs(
    best_pipeline,
    best_model_name,
    comparison,
    X,
    categorical_features,
    numerical_features
):
    """Save trained pipeline and information needed by Flask."""

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    # The complete preprocessing pipeline and classifier are saved together.
    joblib.dump(best_pipeline, MODEL_FILE)

    category_options = {}

    for column in categorical_features:
        category_options[column] = sorted(
            X[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": X.columns.tolist(),
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "category_options": category_options,
        "target_mapping": {
            0: "Rejected",
            1: "Approved"
        },
        "model_comparison": comparison.to_dict(
            orient="records"
        )
    }

    joblib.dump(metadata, METADATA_FILE)

    print("Best model saved as:")
    print(MODEL_FILE)

    print("\nModel metadata saved as:")
    print(METADATA_FILE)

    print("\nModel comparison saved as:")
    print(COMPARISON_FILE)


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():
    try:
        # Load both CSV files.
        application, credit = load_datasets()

        # Display dataset details.
        display_dataset_information(
            application,
            credit
        )

        # Remove exact duplicate records.
        application = application.drop_duplicates()
        credit = credit.drop_duplicates()

        print("\nDuplicates Removed")

        # Create one credit-result label for each applicant.
        credit_summary = create_credit_target(credit)

        # Merge, clean and engineer features.
        data = prepare_dataset(
            application,
            credit_summary
        )

        # Separate independent features and target.
        X = data.drop(columns=["Approved"])
        y = data["Approved"].astype(int)

        if y.nunique() < 2:
            raise ValueError(
                "The target column contains only one class. "
                "Both Approved and Rejected records are required."
            )

        # Create training and testing datasets.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("\n" + "=" * 70)
        print("TRAIN-TEST SPLIT")
        print("=" * 70)

        print("Training Data Shape:", X_train.shape)
        print("Testing Data Shape :", X_test.shape)

        # Create automatic preprocessing.
        (
            preprocessor,
            categorical_features,
            numerical_features
        ) = create_preprocessor(X)

        # Train and compare the algorithms.
        (
            best_pipeline,
            best_model_name,
            comparison
        ) = train_and_compare_models(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )

        # Save trained pipeline and metadata.
        save_outputs(
            best_pipeline,
            best_model_name,
            comparison,
            X,
            categorical_features,
            numerical_features
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nGenerated files:")
        print("1. model.pkl")
        print("2. model_metadata.pkl")
        print("3. model_comparison.csv")

    except FileNotFoundError as error:
        print("\nFILE ERROR:")
        print(error)
        sys.exit(1)

    except Exception as error:
        print("\nTRAINING ERROR:")
        print(type(error).__name__ + ":", error)
        sys.exit(1)


if __name__ == "__main__":
    main()