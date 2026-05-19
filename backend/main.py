from clocktower_img2json.api import create_app

app = create_app(storage_dir="/app/storage")

__all__ = ["app", "create_app"]
