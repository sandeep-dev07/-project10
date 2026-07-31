import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_FILENAME = "placement_predict_50k Dataset (3)(in).csv"
DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_DATA_FILENAME),
)


class DataLoadError(Exception):
    """Raised when the dataset cannot be loaded or is unusable."""


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Dataset is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Dataset could not be parsed as CSV: {path}") from exc
    except OSError as exc:
        raise DataLoadError(f"Dataset could not be read: {path}") from exc

    if df.empty:
        raise DataLoadError(f"Dataset contains no rows: {path}")

    logger.info("Loaded dataset from %s with shape %s", path, df.shape)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    if df is None:
        raise DataLoadError("No DataFrame provided to summarize")

    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_data()
    print(get_data_summary(data))
