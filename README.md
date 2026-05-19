# clocktower-img2json

Converts Blood on the Clocktower script images into script JSON and serves a web dashboard for editing the extracted result.

## Architecture

- `/app/data` stores audit metadata only in SQLite at `/app/data/metadata.db`
- `/app/storage` stores filesystem assets only: `script.json`, `scriptlogo.png`, and cropped token images
- The filesystem copy of `script.json` is the source of truth for script arrays

## Install

```bash
python -m pip install -e .[dev]
```

> `pytesseract` requires the system `tesseract` binary to be installed.

### Optional Google AI Studio OCR helper

If `GEMINI_API_KEY` is set, the converter will try Google AI Studio first using Gemini 3.5 Flash and the normalized PNG image bytes.
If the key is missing, or the Gemini request fails, it automatically falls back to local `pytesseract` OCR.

The following environment variables control Gemini behaviour:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | _(unset)_ | Enable Google AI Studio Gemini OCR. If unset, local OCR is used. |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Vision-capable Gemini model to use. |
| `GEMINI_API_URL` | `https://generativelanguage.googleapis.com/v1beta/models` | Google AI Studio API base URL. |

> **Note:** The configured Gemini model must support image inputs.

## Run the API

```bash
uvicorn clocktower_img2json.api:app --host 0.0.0.0 --port 8000
```

Or from the Docker-oriented wrapper:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000

## Docker

Build:

```bash
docker build -t clocktower-img2json .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/storage:/app/storage" \
  clocktower-img2json
```
```

## Core routes

- `GET /` — upload dashboard
- `GET /script/<uuid>/` — editor dashboard
- `POST /api/upload` — ingest an image, write `/app/storage/<uuid>/script.json`, and create an audit record
- `GET /api/script/<uuid>` — read `script.json` directly from disk and return JSON
- `POST /api/script/<uuid>/update?edited_by=<name>` — overwrite `script.json` and append an edit history row
- `GET /script/<uuid>/scriptlogo.png` — script banner asset
- `GET /script/<uuid>/<asset_name>` — cropped icon assets
