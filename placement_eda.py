import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# =====================================================
# CHANGE THIS PATH TO YOUR DATASET
# =====================================================

CSV_PATH = r"C:\Users\Sandeep\PycharmProjects\Placement_Prediction\placement_predict_50k Dataset (3)(in).csv"


# =====================================================
# SAVE CHARTS
# =====================================================

CHART_FOLDER = os.path.join("static", "charts")
os.makedirs(CHART_FOLDER, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, filename))
    plt.close()


def run_eda():

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n{CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    results = {}

    charts = []

    # =====================================================
    # TASK 1
    # DATA LOADED
    # =====================================================

    results["n_rows"] = data.shape[0]
    results["n_cols"] = data.shape[1]

    results["columns"] = data.columns.tolist()

    results["preview"] = (
        data.head()
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    # =====================================================
    # TASK 2
    # BASIC INFO
    # =====================================================

    results["dtypes"] = (
        data.dtypes
        .astype(str)
        .to_dict()
    )

    numeric = (
        data.describe()
        .round(2)
        .reset_index()
    )

    results["numeric_columns"] = numeric.columns.tolist()

    results["numeric_desc"] = (
        numeric.fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    categorical = (
        data.describe(include="object")
        .reset_index()
    )

    results["categorical_columns"] = categorical.columns.tolist()

    results["categorical_desc"] = (
        categorical.fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    # =====================================================
    # TASK 3
    # MISSING VALUES
    # =====================================================

    missing = data.isnull().sum()

    missing_pct = (missing / len(data)) * 100

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": np.round(missing_pct.values, 2)
    })

    missing_df = missing_df[
        missing_df["Missing Count"] > 0
    ].sort_values(
        "Missing %",
        ascending=False
    )

    results["missing_columns"] = missing_df.columns.tolist()

    results["missing_table"] = (
        missing_df
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    if not missing_df.empty:

        plt.figure(figsize=(10,5), dpi=100)

        sns.barplot(
            x=missing_df["Column"],
            y=missing_df["Missing %"]
        )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Missing %")
        plt.title("Missing Value by Column")

        save_plot("missing_values.png")

        charts.append("missing_values.png")

    # =====================================================
    # TASK 4
    # DUPLICATE ROWS
    # =====================================================

    results["duplicate_count"] = int(
        data.duplicated().sum()
    )

    # =====================================================
    # TASK 5
    # TARGET VARIABLE
    # =====================================================

    results["target_counts"] = (
        data["PlacementStatus"]
        .value_counts()
        .to_dict()
    )

    plt.figure(dpi=125)

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.xlabel(
        "Placement Status (0 = Not Placed, 1 = Placed)"
    )

    plt.ylabel("Count")

    plt.title(
        "Placement Status Distribution"
    )

    save_plot("placement_status.png")

    charts.append("placement_status.png")

    results["charts"] = charts

    return results