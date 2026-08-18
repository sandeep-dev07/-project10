import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# =====================================================
# DATASET PATH
# =====================================================

CSV_PATH = r"C:\Users\Sandeep\PycharmProjects\Placement_Prediction\placement_predict_50k Dataset (3)(in).csv"

# =====================================================
# SAVE CHARTS
# =====================================================

CHART_FOLDER = os.path.join("static", "charts")
os.makedirs(CHART_FOLDER, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_FOLDER, filename)
    )
    plt.close()


def run_eda():

    # =====================================================
    # LOAD DATA
    # =====================================================

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset not found at:\n{CSV_PATH}"
        )

    data = pd.read_csv(CSV_PATH)

    results = {}

    charts = []

    # =====================================================
    # TASK 1 - DATA LOADED
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
    # TASK 2 - BASIC INFO
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

    results["numeric_columns"] = (
        numeric.columns.tolist()
    )

    results["numeric_desc"] = (
        numeric
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    categorical = (
        data.describe(include="object")
        .reset_index()
    )

    results["categorical_columns"] = (
        categorical.columns.tolist()
    )

    results["categorical_desc"] = (
        categorical
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    # =====================================================
    # TASK 3 - MISSING VALUES
    # =====================================================

    missing = data.isnull().sum()

    missing_pct = (
        missing / len(data)
    ) * 100

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": np.round(
            missing_pct.values,
            2
        )
    })

    missing_df = missing_df[
        missing_df["Missing Count"] > 0
    ].sort_values(
        "Missing %",
        ascending=False
    )

    results["missing_columns"] = (
        missing_df.columns.tolist()
    )

    results["missing_table"] = (
        missing_df
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    # -----------------------------------------------------
    # Missing Values Graph
    # EXACT ML_PRO GRAPH
    # -----------------------------------------------------

    if not missing_df.empty:

        plt.figure(
            figsize=(10, 5),
            dpi=100
        )

        sns.barplot(
            x=missing_df["Column"],
            y=missing_df["Missing %"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.ylabel("Missing %")

        plt.title(
            "Missing value by column"
        )

        save_plot(
            "missing_values.png"
        )

        charts.append(
            "missing_values.png"
        )

    # =====================================================
    # TASK 4 - DUPLICATE ROWS
    # =====================================================

    results["duplicate_count"] = int(
        data.duplicated().sum()
    )

    # =====================================================
    # TASK 5 - TARGET VARIABLE
    # PlacementStatus
    # =====================================================

    results["target_counts"] = (
        data["PlacementStatus"]
        .value_counts()
        .to_dict()
    )

    # -----------------------------------------------------
    # Placement Status Graph
    # EXACT ML_PRO GRAPH
    # -----------------------------------------------------

    plt.figure(dpi=125)

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.xlabel(
        "Placement Status (0 = Not Placed, 1 = Placed)"
    )

    plt.ylabel("Counts")

    plt.title(
        "Placement Status Distribution"
    )

    save_plot(
        "placement_status.png"
    )

    charts.append(
        "placement_status.png"
    )

    # =====================================================
    # TASK 6 - NUMERIC FEATURE DISTRIBUTION
    # =====================================================

    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "Publications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "ExtraCurricular",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package"
    ]

    hist_cols = [
        c
        for c in hist_cols
        if c in data.columns
    ]

    if hist_cols:

        data[hist_cols].hist(
            figsize=(14, 10),
            bins=20
        )

        plt.suptitle(
            "Numeric Feature Distribution"
        )

        save_plot(
            "numeric_feature_distribution.png"
        )

        charts.append(
            "numeric_feature_distribution.png"
        )

    # =====================================================
    # TASK 6 - CGPA DISTRIBUTION WITH MEAN
    # =====================================================

    if "CGPA" in data.columns:

        plt.figure(dpi=125)

        sns.histplot(
            data=data["CGPA"],
            kde=True
        )

        plt.axvline(
            x=np.mean(data["CGPA"]),
            color="green",
            linestyle="--",
            label="CGPA"
        )

        plt.legend()

        plt.title(
            "CGPA Distribution with Mean"
        )

        save_plot(
            "cgpa_distribution_mean.png"
        )

        charts.append(
            "cgpa_distribution_mean.png"
        )

    # =====================================================
    # TASK 7 - OUTLIER DETECTION (BOXPLOTS)
    # =====================================================

    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "Publications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "ExtraCurricular",
        "PlacementStatus",
        "IsAnomaly",
        "Salary Package"
    ]

    box_cols = [
        c
        for c in box_cols
        if c in data.columns
    ]

    for col in box_cols:

        plt.figure(
            figsize=(10, 4)
        )

        sns.boxplot(
            x=data[col],
            color="skyblue"
        )

        plt.title(
            f"Boxplot of {col}"
        )

        safe_name = (
            col
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        filename = (
            f"boxplot_{safe_name}.png"
        )

        save_plot(filename)

        charts.append(filename)

    # =====================================================
    # TASK 8 - CORRELATION HEATMAP
    # EXACT ML_PRO GRAPH
    # =====================================================

    corr = (
        data
        .select_dtypes(
            include=[np.number]
        )
        .corr()
    )

    plt.figure(
        figsize=(16, 12),
        dpi=100
    )

    sns.heatmap(
        np.round(corr, 2),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title(
        "Correlation Heatmap"
    )

    save_plot(
        "correlation_heatmap.png"
    )

    charts.append(
        "correlation_heatmap.png"
    )




    # =====================================================
    # TASK 9 - RELATIONSHIP PLOTS
    # =====================================================

    # CGPA vs Salary Package

    if (
        "CGPA" in data.columns
        and "Salary Package" in data.columns
    ):

        plt.figure(
            figsize=(8, 5),
            dpi=120
        )

        sns.regplot(
            data=data,
            x="CGPA",
            y="Salary Package",
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"}
        )

        plt.title(
            "CGPA vs Salary Package"
        )

        save_plot(
            "cgpa_vs_salary.png"
        )

        charts.append(
            "cgpa_vs_salary.png"
        )


    # Aptitude Test Score vs Coding Test Score

    if (
        "AptitudeTestScore" in data.columns
        and "CodingTestScore" in data.columns
    ):

        plt.figure(
            figsize=(8, 5),
            dpi=120
        )

        sns.regplot(
            data=data,
            x="AptitudeTestScore",
            y="CodingTestScore",
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"}
        )

        plt.title(
            "Aptitude Test Score vs Coding Test Score"
        )

        save_plot(
            "aptitude_vs_coding.png"
        )

        charts.append(
            "aptitude_vs_coding.png"
        )


    # =====================================================
    # TASK 10 - CATEGORICAL FEATURE COUNTS
    # =====================================================

    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialization",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [
        c
        for c in cat_cols
        if c in data.columns
    ]

    for col in cat_cols:

        plt.figure(
            figsize=(8, 4),
            dpi=120
        )

        sns.countplot(
            data=data,
            x=col
        )

        plt.xticks(
            rotation=45
        )

        plt.title(
            f"Count Plot - {col}"
        )

        safe_name = (
            col
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
        )

        filename = (
            f"countplot_{safe_name}.png"
        )

        save_plot(filename)

        charts.append(filename)


    # =====================================================
    # TASK 11 - GENDER VS PLACEMENT STATUS
    # =====================================================

    if (
        "Gender" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(
            figsize=(7, 5),
            dpi=120
        )

        sns.countplot(
            data=data,
            x="Gender",
            hue="PlacementStatus"
        )

        plt.title(
            "Gender vs Placement Status"
        )

        save_plot(
            "gender_vs_placement.png"
        )

        charts.append(
            "gender_vs_placement.png"
        )


    # =====================================================
    # TASK 12 - COLLEGE TIER VS PLACEMENT STATUS
    # =====================================================

    if (
        "CollegeTier" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(
            figsize=(8, 5),
            dpi=120
        )

        sns.countplot(
            data=data,
            x="CollegeTier",
            hue="PlacementStatus"
        )

        plt.title(
            "College Tier vs Placement Status"
        )

        save_plot(
            "college_tier_vs_placement.png"
        )

        charts.append(
            "college_tier_vs_placement.png"
        )


    # =====================================================
    # TASK 12 - STREAM VS PLACEMENT STATUS
    # =====================================================

    if (
        "Stream" in data.columns
        and "PlacementStatus" in data.columns
    ):

        plt.figure(
            figsize=(10, 5),
            dpi=120
        )

        sns.countplot(
            data=data,
            x="Stream",
            hue="PlacementStatus"
        )

        plt.xticks(
            rotation=45
        )

        plt.title(
            "Stream vs Placement Status"
        )

        save_plot(
            "stream_vs_placement.png"
        )

        charts.append(
            "stream_vs_placement.png"
        )




    # =====================================================
    # TASK 13 - SGPA TREND ACROSS SEMESTERS
    # =====================================================

    semester_cols = [
        "Sem1_SGPA",
        "Sem2_SGPA",
        "Sem3_SGPA",
        "Sem4_SGPA",
        "Sem5_SGPA",
        "Sem6_SGPA",
        "Sem7_SGPA",
        "Sem8_SGPA"
    ]

    semester_cols = [
        c
        for c in semester_cols
        if c in data.columns
    ]

    if len(semester_cols) > 0:

        avg_sgpa = data[semester_cols].mean()

        plt.figure(
            figsize=(10, 5),
            dpi=120
        )

        plt.plot(
            avg_sgpa.index,
            avg_sgpa.values,
            marker="o",
            linewidth=2
        )

        plt.title(
            "Average SGPA Across Semesters"
        )

        plt.xlabel(
            "Semester"
        )

        plt.ylabel(
            "Average SGPA"
        )

        save_plot(
            "sgpa_trend.png"
        )

        charts.append(
            "sgpa_trend.png"
        )


    # =====================================================
    # TASK 14 - SALARY PACKAGE ANALYSIS
    # =====================================================

    if (
        "Salary Package" in data.columns
        and "PlacementStatus" in data.columns
    ):

        placed = data[
            data["PlacementStatus"] == 1
        ]

        # Salary distribution

        plt.figure(
            figsize=(8, 5),
            dpi=120
        )

        sns.histplot(
            placed["Salary Package"],
            bins=20,
            kde=True
        )

        plt.title(
            "Salary Package Distribution (Placed Students)"
        )

        save_plot(
            "salary_package_distribution.png"
        )

        charts.append(
            "salary_package_distribution.png"
        )


        # Salary by College Tier

        if "CollegeTier" in data.columns:

            plt.figure(
                figsize=(8, 5),
                dpi=120
            )

            sns.boxplot(
                data=placed,
                x="CollegeTier",
                y="Salary Package"
            )

            plt.title(
                "Salary Package by College Tier"
            )

            save_plot(
                "salary_package_by_college_tier.png"
            )

            charts.append(
                "salary_package_by_college_tier.png"
            )


    # =====================================================
    # TASK 15 - PAIRPLOT
    # =====================================================

    pair_cols = [
        "CGPA",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore",
        "PlacementStatus"
    ]

    pair_cols = [
        c
        for c in pair_cols
        if c in data.columns
    ]

    if len(pair_cols) == 5:

        sample_data = data[
            pair_cols
        ].sample(
            n=min(1000, len(data)),
            random_state=42
        )

        pair_plot = sns.pairplot(
            sample_data,
            hue="PlacementStatus",
            diag_kind="hist",
            corner=True
        )

        pair_plot.fig.suptitle(
            "Pairplot of Placement Features",
            y=1.02
        )

        pair_plot.fig.savefig(
            os.path.join(
                CHART_FOLDER,
                "pairplot.png"
            ),
            bbox_inches="tight"
        )

        plt.close("all")

        charts.append(
            "pairplot.png"
        )


    # =====================================================
    # RETURN RESULTS
    # =====================================================

    results["charts"] = charts

    return results