"""
Credit Card Approval Prediction System

Dataset:
    dataset/application_record.csv
    dataset/credit_record.csv

Generated files:
    model.pkl
    model_metadata.pkl
    model_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.base import clone
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

warnings.filterwarnings("ignore", category=FutureWarning)


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

APPLICATION_FILE = DATASET_DIR / "application_record.csv"
CREDIT_FILE = DATASET_DIR / "credit_record.csv"

MODEL_FILE = BASE_DIR / "model.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"
COMPARISON_FILE = BASE_DIR / "model_comparison.csv"


def load_datasets():
    if not APPLICATION_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {APPLICATION_FILE}"
        )

    if not CREDIT_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CREDIT_FILE}"
        )

    application = pd.read_csv(APPLICATION_FILE)
    credit = pd.read_csv(
        CREDIT_FILE,
        dtype={"STATUS": str},
    )

    required_application_columns = {
        "ID",
        "CODE_GENDER",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "FLAG_MOBIL",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "OCCUPATION_TYPE",
        "CNT_FAM_MEMBERS",
    }

    required_credit_columns = {
        "ID",
        "STATUS",
    }

    missing_application = required_application_columns.difference(
        application.columns
    )
    missing_credit = required_credit_columns.difference(
        credit.columns
    )

    if missing_application:
        raise ValueError(
            "Application dataset is missing columns: "
            + ", ".join(sorted(missing_application))
        )

    if missing_credit:
        raise ValueError(
            "Credit dataset is missing columns: "
            + ", ".join(sorted(missing_credit))
        )

    print("Application shape:", application.shape)
    print("Credit shape     :", credit.shape)

    return application, credit


def create_credit_target(credit):
    credit = credit.copy()
    credit["STATUS"] = credit["STATUS"].astype(str).str.strip()

    allowed_statuses = {"X", "C", "0", "1", "2", "3", "4", "5"}
    invalid_statuses = set(credit["STATUS"].dropna().unique()) - allowed_statuses

    if invalid_statuses:
        print(
            "Warning: Unknown credit statuses found:",
            sorted(invalid_statuses),
        )

    good_statuses = {"X", "C", "0"}

    credit["Approved"] = credit["STATUS"].isin(
        good_statuses
    ).astype(int)

    return (
        credit.groupby("ID", as_index=False)["Approved"]
        .min()
    )


def prepare_dataset(application, credit_summary):
    application = (
        application.copy()
        .drop_duplicates(subset=["ID"])
    )

    data = pd.merge(
        application,
        credit_summary,
        on="ID",
        how="inner",
        validate="one_to_one",
    )

    if data.empty:
        raise ValueError(
            "No matching applicant IDs were found between the datasets."
        )

    data = data.drop(columns=["ID"])

    data["AGE"] = (
        -pd.to_numeric(data["DAYS_BIRTH"], errors="coerce")
        / 365.25
    ).round()

    employed_days = pd.to_numeric(
        data["DAYS_EMPLOYED"],
        errors="coerce",
    )

    data["EMPLOYMENT_YEARS"] = np.where(
        employed_days.eq(365243),
        0.0,
        -employed_days / 365.25,
    )

    data["EMPLOYMENT_YEARS"] = (
        pd.Series(data["EMPLOYMENT_YEARS"], index=data.index)
        .clip(lower=0)
        .round(1)
    )

    data = data.drop(
        columns=["DAYS_BIRTH", "DAYS_EMPLOYED"]
    )

    data["OCCUPATION_TYPE"] = (
        data["OCCUPATION_TYPE"]
        .fillna("Unknown")
        .astype(str)
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.drop_duplicates()

    numeric_columns = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for column in numeric_columns:
        if column != "Approved":
            median = data[column].median()
            data[column] = data[column].fillna(median)

    categorical_columns = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    data["AGE"] = data["AGE"].clip(18, 100).astype(int)
    data["CNT_CHILDREN"] = data["CNT_CHILDREN"].clip(lower=0)
    data["CNT_FAM_MEMBERS"] = data["CNT_FAM_MEMBERS"].clip(lower=1)
    data["AMT_INCOME_TOTAL"] = data["AMT_INCOME_TOTAL"].clip(lower=1)
    data["EMPLOYMENT_YEARS"] = data["EMPLOYMENT_YEARS"].clip(lower=0)

    print("Prepared shape:", data.shape)
    print("Target distribution:")
    print(data["Approved"].value_counts())

    return data


def create_preprocessor(X):
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("scaler", StandardScaler())
                    ]
                ),
                numerical_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        )
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return (
        preprocessor,
        categorical_features,
        numerical_features,
    )


def create_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        print("XGBoost is unavailable and will be skipped.")

    return models


def evaluate_predictions(y_test, predictions):
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test,
            predictions,
        ),
        "Macro F1 Score": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "Rejected Precision": precision_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "Rejected Recall": recall_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
        "Rejected F1 Score": f1_score(
            y_test,
            predictions,
            pos_label=0,
            zero_division=0,
        ),
    }


def train_and_compare_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):
    results = []
    trained_pipelines = {}

    for model_name, classifier in create_models().items():
        print("\nTraining:", model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", classifier),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            metrics = evaluate_predictions(
                y_test,
                predictions,
            )
            metrics["Model"] = model_name

            results.append(metrics)
            trained_pipelines[model_name] = pipeline

            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=["Rejected", "Approved"],
                    zero_division=0,
                )
            )
            print("Confusion matrix:")
            print(confusion_matrix(y_test, predictions))

        except Exception as error:
            print(
                f"{model_name} failed: "
                f"{type(error).__name__}: {error}"
            )

    if not results:
        raise RuntimeError(
            "No model was trained successfully."
        )

    comparison = pd.DataFrame(results)[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
            "Macro F1 Score",
            "Rejected Precision",
            "Rejected Recall",
            "Rejected F1 Score",
        ]
    ]

    comparison = comparison.sort_values(
        by=[
            "Macro F1 Score",
            "Balanced Accuracy",
            "Rejected F1 Score",
        ],
        ascending=False,
    ).reset_index(drop=True)

    comparison.to_csv(COMPARISON_FILE, index=False)

    best_model_name = str(comparison.iloc[0]["Model"])
    best_pipeline = trained_pipelines[best_model_name]

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print("\nBest model:", best_model_name)

    return best_pipeline, best_model_name, comparison


def save_outputs(
    best_pipeline,
    best_model_name,
    comparison,
    X,
    categorical_features,
    numerical_features,
):
    joblib.dump(best_pipeline, MODEL_FILE)

    category_options = {
        column: sorted(
            X[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        for column in categorical_features
    }

    metadata = {
        "best_model_name": best_model_name,
        "feature_columns": X.columns.tolist(),
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "category_options": category_options,
        "target_mapping": {
            0: "Rejected",
            1: "Approved",
        },
        "model_comparison": comparison.to_dict(
            orient="records"
        ),
        "scikit_learn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    joblib.dump(metadata, METADATA_FILE)

    print("\nSaved:")
    print(MODEL_FILE)
    print(METADATA_FILE)
    print(COMPARISON_FILE)


def main():
    try:
        application, credit = load_datasets()

        application = application.drop_duplicates()
        credit = credit.drop_duplicates()

        credit_summary = create_credit_target(credit)
        data = prepare_dataset(
            application,
            credit_summary,
        )

        X = data.drop(columns=["Approved"])
        y = data["Approved"].astype(int)

        if y.nunique() < 2:
            raise ValueError(
                "The target contains only one class."
            )

        class_counts = y.value_counts()

        if class_counts.min() < 2:
            raise ValueError(
                "Each target class must contain at least two records."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

        (
            preprocessor,
            categorical_features,
            numerical_features,
        ) = create_preprocessor(X)

        (
            best_pipeline,
            best_model_name,
            comparison,
        ) = train_and_compare_models(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )

        save_outputs(
            best_pipeline,
            best_model_name,
            comparison,
            X,
            categorical_features,
            numerical_features,
        )

        print("\nTRAINING COMPLETED SUCCESSFULLY")

    except Exception as error:
        print(
            f"\nERROR: {type(error).__name__}: {error}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
