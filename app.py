import importlib
from pathlib import Path

import pandas as pd
from flask import Flask, render_template

from decision_tree_model import train_decision_tree_model
from linear_regression_sklearn import train_linear_regression_model
from load_data import get_data_summary
from logistic_regression import train_logistic_regression_model
from one_hot_encoding import preprocess_dataframe
from placement_eda import run_eda

app = Flask(__name__)


def get_preprocessing_summary() -> dict:
    csv_path = Path(__file__).resolve().parent / "placement_predict_50k Dataset (3)(in).csv"
    df = pd.read_csv(csv_path)
    processed_df = preprocess_dataframe(df)

    return {
        "original_rows": int(len(df)),
        "processed_rows": int(len(processed_df)),
        "original_columns": list(df.columns),
        "processed_columns": list(processed_df.columns),
        "missing_before": int(df.isna().sum().sum()),
        "missing_after": int(processed_df.isna().sum().sum()),
        "target_column": "PlacementStatus",
        "preview": processed_df.head(5).to_dict("records"),
    }


def get_boosting_model_results():
    """Run all boosting models together and return the successful summaries plus any errors."""
    modules = [
        ("adaboost", "AdaBoost", "ada_boosting"),
        ("gradient-boosting", "Gradient Boosting", "gradient_boosting_machine"),
        ("xgboost", "XGBoost", "xgboost_model"),
        ("lightgbm", "LightGBM", "lightgbm_model"),
        ("catboost", "CatBoost", "catboost_model"),
        ("hist-gradient-boosting", "HistGradientBoosting", "hist_gradient_boosting"),
        ("logitboost", "LogitBoost", "logitboost_model"),
    ]

    summaries = []
    errors = []
    for slug, title, module_name in modules:
        try:
            module = importlib.import_module(module_name)
            result = module.train_model()
            summaries.append({"slug": slug, "title": title, "result": result})
        except Exception as exc:  # pragma: no cover - optional dependency failure should stay visible
            errors.append({"slug": slug, "title": title, "error": str(exc)})

    return summaries, errors


@app.route("/")
def index():
    # Landing page, no section selected yet
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    """Loads the dataset (server-side) and renders the summary into the page."""
    error = None
    summary = None
    try:
        summary = get_data_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )


@app.route("/preprocessing")
def preprocessing():
    """Shows the preprocessing result using the one-hot encoding workflow."""
    error = None
    summary = None
    try:
        summary = get_preprocessing_summary()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="preprocessing",
        preprocessing_summary=summary,
        error=error,
    )


@app.route("/linear-regression")
def linear_regression():
    """Trains and displays a linear regression model summary for salary prediction."""
    error = None
    model_summary = None
    try:
        model_summary = train_linear_regression_model()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="linear-regression",
        model_summary=model_summary,
        error=error,
    )


@app.route("/logistic-regression")
def logistic_regression():
    """Trains and displays a logistic regression model summary for placement prediction."""
    error = None
    model_summary = None
    try:
        model_summary = train_logistic_regression_model()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="logistic-regression",
        logistic_summary=model_summary,
        error=error,
    )


@app.route("/decision-tree")
def decision_tree():
    """Displays the decision tree and all boosting techniques on one dashboard page."""
    error = None
    model_summary = None
    boosting_summaries = []
    boosting_errors = []

    try:
        model_summary = train_decision_tree_model()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    try:
        boosting_summaries, boosting_errors = get_boosting_model_results()
    except Exception as e:
        boosting_errors = [{"slug": "dashboard", "title": "Boosting models", "error": str(e)}]

    return render_template(
        "index.html",
        active="decision-tree",
        decision_tree_summary=model_summary,
        boosting_summaries=boosting_summaries,
        boosting_errors=boosting_errors,
        error=error,
    )


@app.route("/eda")
def eda():
    """Runs exploratory data analysis and renders results."""
    error = None
    eda_output = None
    try:
        eda_output = run_eda()   # call your EDA function
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=eda_output,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
