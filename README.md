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

## Run the API

```bash
uvicorn clocktower_img2json.api:app --host 0.0.0.0 --port 8000
```

Or from the Docker-oriented wrapper:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Core routes

- `GET /` — upload dashboard
- `GET /dashboard/edit.html?id=<uuid>` — editor dashboard
- `POST /api/upload` — ingest an image, write `/app/storage/<uuid>/script.json`, and create an audit record
- `GET /api/script/<uuid>` — read `script.json` directly from disk and return JSON
- `POST /api/script/<uuid>/update?edited_by=<name>` — overwrite `script.json` and append an edit history row
- `GET /script/<uuid>/scriptlogo.png` — script banner asset
- `GET /script/<uuid>/<asset_name>` — cropped icon assets
