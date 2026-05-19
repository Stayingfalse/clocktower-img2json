from __future__ import annotations

import argparse
import json
from pathlib import Path

from .converter import convert_image_to_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Clocktower script image to JSON")
    parser.add_argument("--image-url", required=True, help="Image URL to process")
    parser.add_argument("--output-dir", default="storage", help="Directory for generated files")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Public base URL for generated image links")
    parser.add_argument("--script-name", default=None, help="Override script name")
    parser.add_argument("--author", default=None, help="Override script author")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_image_to_script(
        image_url=args.image_url,
        storage_dir=Path(args.output_dir).resolve(),
        public_base_url=args.base_url,
        script_name_override=args.script_name,
        author_override=args.author,
    )

    print(json.dumps(
        {
            "uuid": result.request_id,
            "script_path": str(result.script_path),
            "source_image": str(result.image_path),
            "homebrew_images": result.image_urls,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
