import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


logger = logging.getLogger(__name__)


def get_ui_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "ui"


def register(app: FastAPI):
    ui_dir = get_ui_dir()
    if not ui_dir.is_dir():
        logger.warning("UI assets directory is missing; skipping UI route registration: %s", ui_dir)
        return

    for name in ["css", "js", "static"]:
        app.mount(
            f"/{name}",
            StaticFiles(directory=ui_dir / name),
            name=name,
        )

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ui_dir / "index.html")
