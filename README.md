# clocktower-img2json

Converts Blood on the Clocktower script images into script JSON compatible with the app schema.

## Features
- Accepts uploaded or local script image files
- OCR extraction of script title, author, role names, and ability text
- Official roles are emitted as role ID strings
- Homebrew roles are emitted as full objects and get icon cutouts from the uploaded image
- Every request gets a UUID and output is stored in `storage/<uuid>/`
- Homebrew icon URLs are UUID-scoped (`/assets/<uuid>/images/<role-id>.png`)
- Output is validated against `script-schema.json` from The Pandemonium Institute repo

## Install

```bash
python -m pip install -e .[dev]
```

> `pytesseract` requires the system `tesseract` binary to be installed.

## CLI usage

```bash
clocktower-img2json \
  --image-file "/absolute/path/to/script.png" \
  --output-dir "/absolute/path/to/storage" \
  --base-url "http://localhost:8000"
```

## API usage

Run the API:

```bash
uvicorn clocktower_img2json.api:app --host 0.0.0.0 --port 8000
```

Convert image:

```bash
curl -X POST http://localhost:8000/scripts/from-upload \
  -F "image=@/absolute/path/to/script.png" \
  -F "script_name=Optional Script Name" \
  -F "author=Optional Author"
```

Response includes:
- `uuid`
- `json_url` (direct link to generated JSON)
- `source_image_url`
- `homebrew_images` map with direct image URLs
- generated `script`
