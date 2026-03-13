# Selling Backend

This project is a Vision Agents backend for a product-listing assistant. It connects to Stream, guides a user through capturing a product screenshot from a live video feed, can polish that image with Nano Banana, and generates product details with Google search + Gemini.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- Access to the local `Vision-Agents` repositories referenced by `pyproject.toml`

## Setup

1. Create an environment file from the example:

```bash
cp .env.example .env
```

2. Fill in the values in `.env`:

```env
# Stream API credentials.
STREAM_API_KEY=
STREAM_API_SECRET=

# Google API credentials. These need to be the same.
GOOGLE_API_KEY=
NANO_BANANA_API_KEY=
```

Notes:
- `STREAM_API_KEY` and `STREAM_API_SECRET` are used for the Stream connection.
- `GOOGLE_API_KEY` is used for product research via the Google/Gemini integration.
- `NANO_BANANA_API_KEY` is used for image polishing. The example indicates it should match `GOOGLE_API_KEY`.

3. Install dependencies:

```bash
uv sync
```

## Run The Project

Start the backend server with:

```bash
uv run python main.py serve --http-log-level=DEBUG
```

If you already created the virtual environment with `uv sync`, this also works:

```bash
.venv/bin/python main.py serve --http-log-level=DEBUG
```

## What The App Does

The runtime workflow is:

1. Join a Stream-powered session.
2. Detect and capture a clear product image from the user's video feed.
3. Generate a polished product image.
4. Research the product and generate structured product details.
5. Emit workflow progress and results back through Stream events.

## Optional

Run the tests with:

```bash
uv run pytest
```
