import logging
import os

from flask import Flask, render_template

from load_data import load_data, get_data_summary

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route("/")
def index():
    return render_template("index.html", active="none")

@app.route("/data-loading")
def data_loading():
    error = None
    summary = None

    try:
        df = load_data()          # Load CSV
        summary = get_data_summary(df)   # Pass DataFrame
    except FileNotFoundError:
        logger.exception("Dataset file is missing")
        error = "Dataset could not be found. Check the server configuration."
    except Exception:
        logger.exception("Failed to load dataset")
        error = "An unexpected error occurred while loading the dataset."

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
    )

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")),
            debug=debug)
