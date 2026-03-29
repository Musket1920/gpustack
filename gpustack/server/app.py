import asyncio
import contextlib
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import aiohttp
from fastapi import FastAPI
from fastapi_cdn_host import patch_docs

from gpustack import __version__
from gpustack.api import exceptions, middlewares
from gpustack.config.config import Config
from gpustack import envs
from gpustack.routes import ui
from gpustack.routes.routes import api_router
from gpustack.server.worker_command_controller import WorkerCommandController
from gpustack.server.worker_control import worker_control_session_registry
from gpustack.utils.forwarded import ForwardedHostPortMiddleware
from gpustack.gateway.plugins import register as register_plugins


logger = logging.getLogger(__name__)


def create_app(cfg: Config) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        connector = aiohttp.TCPConnector(
            limit=envs.TCP_CONNECTOR_LIMIT,
            force_close=True,
        )
        worker_command_controller_task = None
        app.state.http_client = aiohttp.ClientSession(
            connector=connector, trust_env=True
        )
        app.state.http_client_no_proxy = aiohttp.ClientSession(connector=connector)
        worker_command_controller_task = asyncio.create_task(
            WorkerCommandController().start()
        )
        try:
            yield
        finally:
            if worker_command_controller_task is not None:
                worker_command_controller_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker_command_controller_task
            await app.state.http_client.close()
            await app.state.http_client_no_proxy.close()

    app = FastAPI(
        title="GPUStack",
        lifespan=lifespan,
        response_model_exclude_unset=True,
        version=__version__,
        docs_url=None if (cfg and cfg.disable_openapi_docs) else "/docs",
        redoc_url=None if (cfg and cfg.disable_openapi_docs) else "/redoc",
        openapi_url=None if (cfg and cfg.disable_openapi_docs) else "/openapi.json",
    )
    app.state.server_config = cfg
    app.state.worker_control_session_registry = worker_control_session_registry
    ui_static_dir = ui.get_ui_dir() / "static"
    if ui_static_dir.is_dir():
        patch_docs(app, ui_static_dir)
    else:
        logger.warning(
            "UI static assets directory is missing; skipping docs asset patching: %s",
            ui_static_dir,
        )
    app.add_middleware(ForwardedHostPortMiddleware)
    app.add_middleware(middlewares.RequestTimeMiddleware)
    app.add_middleware(middlewares.ModelUsageMiddleware)
    app.add_middleware(middlewares.RefreshTokenMiddleware)
    app.include_router(api_router)
    ui.register(app)
    register_plugins(cfg=cfg, app=app)
    exceptions.register_handlers(app)

    return app
