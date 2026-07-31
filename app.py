from flask import Flask, render_template

from load_data import get_data_summary, load_data

app = Flask(__name__)


def render_page(active: str, **context):
    """Render the single-page shell with the given section active."""
    return render_template("index.html", active=active, **context)


def safe_call(func, *args, **kwargs):
    """Run ``func`` and return ``(result, error_message)`` instead of raising."""
    try:
        return func(*args, **kwargs), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error: {e}"


@app.route("/")
def index():
    return render_page("none")


@app.route("/data-loading")
def data_loading():
    df, error = safe_call(load_data)
    summary = None
    if error is None:
        summary, error = safe_call(get_data_summary, df)

    return render_page("data-loading", summary=summary, error=error)


if __name__ == "__main__":
    app.run(debug=True)
