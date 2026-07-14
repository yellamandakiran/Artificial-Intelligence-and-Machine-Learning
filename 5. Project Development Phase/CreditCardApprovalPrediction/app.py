from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.pkl"
METADATA_FILE = BASE_DIR / "model_metadata.pkl"


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD MODEL AND METADATA
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "model.pkl was not found. Run train_model.py first."
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        "model_metadata.pkl was not found. Run train_model.py first."
    )


model = joblib.load(MODEL_FILE)
metadata = joblib.load(METADATA_FILE)

FEATURE_COLUMNS = metadata["feature_columns"]
CATEGORY_OPTIONS = metadata["category_options"]
BEST_MODEL_NAME = metadata["best_model_name"]


# ============================================================
# INPUT CONVERSION
# ============================================================

def convert_checkbox(value):
    """
    Convert checkbox, Yes/No and Y/N values into 1 or 0.
    """

    if value is None:
        return 0

    value = str(value).strip().lower()

    return 1 if value in {
        "1",
        "true",
        "yes",
        "y",
        "on"
    } else 0


def create_input_dataframe(form):
    """
    Convert values from the HTML form into a Pandas DataFrame.

    The column names and order must exactly match the columns
    used during model training.
    """

    applicant_data = {
        "CODE_GENDER": form.get("CODE_GENDER", "M"),
        "FLAG_OWN_CAR": form.get("FLAG_OWN_CAR", "N"),
        "FLAG_OWN_REALTY": form.get("FLAG_OWN_REALTY", "N"),

        "CNT_CHILDREN": int(
            form.get("CNT_CHILDREN", 0)
        ),

        "AMT_INCOME_TOTAL": float(
            form.get("AMT_INCOME_TOTAL", 0)
        ),

        "NAME_INCOME_TYPE": form.get(
            "NAME_INCOME_TYPE",
            "Working"
        ),

        "NAME_EDUCATION_TYPE": form.get(
            "NAME_EDUCATION_TYPE",
            "Secondary / secondary special"
        ),

        "NAME_FAMILY_STATUS": form.get(
            "NAME_FAMILY_STATUS",
            "Single / not married"
        ),

        "NAME_HOUSING_TYPE": form.get(
            "NAME_HOUSING_TYPE",
            "House / apartment"
        ),

        "FLAG_MOBIL": convert_checkbox(
            form.get("FLAG_MOBIL")
        ),

        "FLAG_WORK_PHONE": convert_checkbox(
            form.get("FLAG_WORK_PHONE")
        ),

        "FLAG_PHONE": convert_checkbox(
            form.get("FLAG_PHONE")
        ),

        "FLAG_EMAIL": convert_checkbox(
            form.get("FLAG_EMAIL")
        ),

        "OCCUPATION_TYPE": form.get(
            "OCCUPATION_TYPE",
            "Unknown"
        ),

        "CNT_FAM_MEMBERS": float(
            form.get("CNT_FAM_MEMBERS", 1)
        ),

        "AGE": int(
            form.get("AGE", 18)
        ),

        "EMPLOYMENT_YEARS": float(
            form.get("EMPLOYMENT_YEARS", 0)
        )
    }

    input_data = pd.DataFrame(
        [applicant_data]
    )

    # Ensure the same column order used during training.
    input_data = input_data[FEATURE_COLUMNS]

    return input_data


# ============================================================
# VALIDATION
# ============================================================

def validate_form(form):
    """
    Validate the numerical form values.
    Returns an error message when invalid.
    """

    try:
        age = int(form.get("AGE", 0))
        income = float(form.get("AMT_INCOME_TOTAL", 0))
        children = int(form.get("CNT_CHILDREN", 0))
        family_members = float(
            form.get("CNT_FAM_MEMBERS", 0)
        )
        employment_years = float(
            form.get("EMPLOYMENT_YEARS", 0)
        )

    except ValueError:
        return "Please enter valid numerical values."

    if age < 18 or age > 100:
        return "Age must be between 18 and 100."

    if income <= 0:
        return "Annual income must be greater than zero."

    if children < 0:
        return "Number of children cannot be negative."

    if family_members < 1:
        return "Family members must be at least 1."

    if employment_years < 0:
        return "Employment years cannot be negative."

    if employment_years > age - 14:
        return (
            "Employment years cannot be greater than "
            "the applicant's possible working age."
        )

    return None


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        model_name=BEST_MODEL_NAME
    )


@app.route("/predict", methods=["GET"])
def prediction_form():
    return render_template(
        "predict.html",
        category_options=CATEGORY_OPTIONS
    )


@app.route("/predict", methods=["POST"])
def predict():
    error_message = validate_form(request.form)

    if error_message:
        return render_template(
            "predict.html",
            category_options=CATEGORY_OPTIONS,
            error=error_message,
            form_data=request.form
        )

    try:
        input_data = create_input_dataframe(
            request.form
        )

        prediction = int(
            model.predict(input_data)[0]
        )

        approval_probability = None
        rejection_probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = list(
                model.classes_
            )

            if 1 in classes:
                approved_index = classes.index(1)

                approval_probability = round(
                    float(
                        probabilities[approved_index]
                    ) * 100,
                    2
                )

            if 0 in classes:
                rejected_index = classes.index(0)

                rejection_probability = round(
                    float(
                        probabilities[rejected_index]
                    ) * 100,
                    2
                )

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
            approval_probability=approval_probability,
            rejection_probability=rejection_probability,
            model_name=BEST_MODEL_NAME
        )

    except Exception as error:
        print("Prediction Error:", error)

        return render_template(
            "predict.html",
            category_options=CATEGORY_OPTIONS,
            error=(
                "The prediction could not be completed. "
                "Please verify all input values."
            ),
            form_data=request.form
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )