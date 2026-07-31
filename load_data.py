import os
from typing import Any, Iterable, Sequence

import pandas as pd

DATA_PATH = r"C:\Users\Sandeep\PycharmProjects\MACHINELEARNING\placement_predict_50k Dataset (3)(in).csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def build_table_rows(
    records: Iterable[dict], columns: Sequence[str]
) -> list[list[Any]]:
    """Turn record dicts into row lists ordered by ``columns``."""
    return [[record[col] for col in columns] for record in records]


def get_data_summary(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    column_info = [
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isnull().sum()),
        }
        for col in columns
    ]
    preview = df.head(10).to_dict(orient="records")

    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": columns,
        "column_rows": build_table_rows(column_info, ["column", "dtype", "missing"]),
        "preview_rows": build_table_rows(preview, columns),
    }


if __name__ == "__main__":
    data = load_data()
    print(get_data_summary(data))
