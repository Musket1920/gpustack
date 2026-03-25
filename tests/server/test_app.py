import logging
import sys
import types

sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))

from gpustack.routes import ui
from gpustack.server.app import create_app


def test_create_app_with_vendored_ui_assets(config):
    ui_dir = ui.get_ui_dir()

    assert (ui_dir / "index.html").is_file()
    for name in ["css", "js", "static"]:
        assert (ui_dir / name).is_dir()

    app = create_app(config)

    route_paths = [getattr(route, "path", None) for route in app.routes]
    route_names = {getattr(route, "name", None) for route in app.routes}

    assert "/" in route_paths
    assert {"css", "js", "static"}.issubset(route_names)


def test_create_app_without_ui_assets(monkeypatch, config, tmp_path, caplog):
    missing_ui_dir = tmp_path / "missing-ui"

    monkeypatch.setattr("gpustack.routes.ui.get_ui_dir", lambda: missing_ui_dir)

    with caplog.at_level(logging.WARNING):
        app = create_app(config)

    route_paths = [getattr(route, "path", None) for route in app.routes]

    assert app is not None
    assert "/openapi.json" in route_paths
    assert "/" not in route_paths
    assert "skipping docs asset patching" in caplog.text
    assert "skipping UI route registration" in caplog.text
