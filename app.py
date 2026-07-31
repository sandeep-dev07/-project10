import logging

from flask import Flask, render_template

from load_data import DataLoadError, get_data_summary, load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    status = 200

    try:
        df = load_data()
        summary = get_data_summary(df)
    except FileNotFoundError as exc:
        logger.exception("Dataset file is missing")
        error = str(exc)
        status = 404
    except DataLoadError as exc:
        logger.exception("Dataset could not be loaded")
        error = str(exc)
        status = 500
    except Exception:
        logger.exception("Unexpected error while loading dataset")
        error = "An unexpected error occurred while loading the dataset. See server logs for details."
        status = 500

    return (
        render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            error=error,
        ),
        status,
    )


if __name__ == "__main__":
    app.run(debug=True)
