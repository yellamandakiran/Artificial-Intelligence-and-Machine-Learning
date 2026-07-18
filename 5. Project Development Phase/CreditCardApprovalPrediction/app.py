from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"

app = Flask(__name__)


def load_artifacts():
    missing = [
        path.name
        for path in (MODEL_FILE, METADATA_FILE)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Run: python train_model.py"
        )

    try:
        trained_model = joblib.load(MODEL_FILE)
        model_metadata = joblib.load(METADATA_FILE)
    except Exception as error:
        raise RuntimeError(
            "The model files could not be loaded. "
            "Delete model.pkl and model_metadata.pkl, "
            "then run train_model.py again."
        ) from error

    if not hasattr(trained_model, "predict"):
        raise TypeError("model.pkl does not contain a valid trained model.")

    if not isinstance(model_metadata, dict):
        raise TypeError("model_metadata.pkl must contain a dictionary.")

    required_keys = {
        "feature_columns",
        "category_options",
        "best_model_name",
    }

    missing_keys = required_keys.difference(model_metadata)

    if missing_keys:
        raise KeyError(
            "Missing metadata key(s): "
            + ", ".join(sorted(missing_keys))
        )

    return trained_model, model_metadata


model, metadata = load_artifacts()

FEATURE_COLUMNS = list(metadata["feature_columns"])
CATEGORY_OPTIONS = dict(metadata["category_options"])
BEST_MODEL_NAME = str(metadata["best_model_name"])


def to_binary(value: Any) -> int:
    if value is None:
        return 0

    return int(
        str(value).strip().lower()
        in {"1", "true", "yes", "y", "on"}
    )


def read_int(
    form: Mapping[str, Any],
    field_name: str,
    default: int = 0,
) -> int:
    value = form.get(field_name, default)

    if value is None or str(value).strip() == "":
        return default

    number = float(str(value).strip())

    if not number.is_integer():
        raise ValueError(f"{field_name} must be a whole number.")

    return int(number)


def read_float(
    form: Mapping[str, Any],
    field_name: str,
    default: float = 0.0,
) -> float:
    value = form.get(field_name, default)

    if value is None or str(value).strip() == "":
        return default

    return float(str(value).strip())


def validate_form(form: Mapping[str, Any]) -> str | None:
    try:
        age = read_int(form, "AGE")
        income = read_float(form, "AMT_INCOME_TOTAL")
        children = read_int(form, "CNT_CHILDREN")
        family_members = read_float(form, "CNT_FAM_MEMBERS")
        employment_years = read_float(form, "EMPLOYMENT_YEARS")
    except (TypeError, ValueError, OverflowError) as error:
        return str(error) or "Please enter valid numerical values."

    if not 18 <= age <= 100:
        return "Age must be between 18 and 100."

    if income <= 0:
        return "Annual income must be greater than zero."

    if children < 0:
        return "Number of children cannot be negative."

    if family_members < 1:
        return "Family members must be at least 1."

    if family_members < children + 1:
        return (
            "Family members must include the applicant "
            "and all children."
        )

    if employment_years < 0:
        return "Employment years cannot be negative."

    if employment_years > max(age - 14, 0):
        return (
            "Employment years cannot be greater than "
            "the applicant's possible working age."
        )

    for field_name, allowed_values in CATEGORY_OPTIONS.items():
        submitted = str(form.get(field_name, "")).strip()

        if submitted and submitted not in allowed_values:
            return f"Please select a valid value for {field_name}."

    return None


def create_input_dataframe(form: Mapping[str, Any]) -> pd.DataFrame:
    applicant = {
        "CODE_GENDER": str(form.get("CODE_GENDER", "M")).strip(),
        "FLAG_OWN_CAR": str(form.get("FLAG_OWN_CAR", "N")).strip(),
        "FLAG_OWN_REALTY": str(form.get("FLAG_OWN_REALTY", "N")).strip(),
        "CNT_CHILDREN": read_int(form, "CNT_CHILDREN"),
        "AMT_INCOME_TOTAL": read_float(form, "AMT_INCOME_TOTAL"),
        "NAME_INCOME_TYPE": str(
            form.get("NAME_INCOME_TYPE", "Working")
        ).strip(),
        "NAME_EDUCATION_TYPE": str(
            form.get(
                "NAME_EDUCATION_TYPE",
                "Secondary / secondary special",
            )
        ).strip(),
        "NAME_FAMILY_STATUS": str(
            form.get(
                "NAME_FAMILY_STATUS",
                "Single / not married",
            )
        ).strip(),
        "NAME_HOUSING_TYPE": str(
            form.get(
                "NAME_HOUSING_TYPE",
                "House / apartment",
            )
        ).strip(),
        "FLAG_MOBIL": to_binary(form.get("FLAG_MOBIL")),
        "FLAG_WORK_PHONE": to_binary(form.get("FLAG_WORK_PHONE")),
        "FLAG_PHONE": to_binary(form.get("FLAG_PHONE")),
        "FLAG_EMAIL": to_binary(form.get("FLAG_EMAIL")),
        "OCCUPATION_TYPE": str(
            form.get("OCCUPATION_TYPE", "Unknown")
        ).strip(),
        "CNT_FAM_MEMBERS": read_float(
            form,
            "CNT_FAM_MEMBERS",
            1.0,
        ),
        "AGE": read_int(form, "AGE", 18),
        "EMPLOYMENT_YEARS": read_float(
            form,
            "EMPLOYMENT_YEARS",
            0.0,
        ),
    }

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in applicant
    ]

    if missing_features:
        raise ValueError(
            "The form is missing trained feature(s): "
            + ", ".join(missing_features)
        )

    return pd.DataFrame(
        [[applicant[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )


def get_probabilities(input_data: pd.DataFrame):
    if not hasattr(model, "predict_proba"):
        return None, None

    probabilities = model.predict_proba(input_data)[0]
    classes = list(getattr(model, "classes_", []))

    approval = None
    rejection = None

    if 1 in classes:
        approval = round(
            float(probabilities[classes.index(1)]) * 100,
            2,
        )

    if 0 in classes:
        rejection = round(
            float(probabilities[classes.index(0)]) * 100,
            2,
        )

    return approval, rejection


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=BEST_MODEL_NAME,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template(
            "predict.html",
            category_options=CATEGORY_OPTIONS,
        )

    error_message = validate_form(request.form)

    if error_message:
        return render_template(
            "predict.html",
            category_options=CATEGORY_OPTIONS,
            error=error_message,
            form_data=request.form,
        )

    try:
        input_data = create_input_dataframe(request.form)
        prediction = int(model.predict(input_data)[0])
        approval, rejection = get_probabilities(input_data)

        if prediction == 1:
            result = "Credit Card Application Approved"
            status = "approved"
        else:
            result = "Credit Card Application Rejected"
            status = "rejected"

        return render_template(
            "result.html",
            prediction=result,
            status=status,
            approval_probability=approval,
            rejection_probability=rejection,
            model_name=BEST_MODEL_NAME,
        )

    except Exception:
        app.logger.exception("Prediction failed")

        return render_template(
            "predict.html",
            category_options=CATEGORY_OPTIONS,
            error=(
                "Prediction failed. Run train_model.py again "
                "and restart the application."
            ),
            form_data=request.form,
        )


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": BEST_MODEL_NAME,
        "feature_count": len(FEATURE_COLUMNS),
    }


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )
