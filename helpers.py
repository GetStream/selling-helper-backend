from datetime import datetime


def get_date_and_time_now() -> str:
    """Return a dash-separated timestamp suitable for filenames, e.g. 2024-04-25-13-54-03."""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
