from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "placement_predict_50k Dataset (3)(in).csv"


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = DATA_PATH

    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (BASE_DIR / file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_csv(file_path)


def get_data_summary() -> dict:
    df = load_data()
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict("records"),
    }
    return summary


if __name__ == "__main__":
    print(get_data_summary())
