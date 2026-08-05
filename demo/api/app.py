"""FastAPI application: read-only API plus the static dashboard.

The API never writes to the database and never runs inference — the consumer
process does both. Keeping them separate means an HTTP handler can never stall
the Kafka poll loop, and restarting the UI mid-demo does not lose stream
position.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from demo.api import routes_alerts, routes_health, routes_method, routes_stamps
from demo.config import get_settings

log = logging.getLogger("demo.api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Real-time multimodal transient classification — demo",
        description=(
            "Live ZTF alert classification into SN / AGN / VS by late fusion of "
            "a LightGBM light-curve branch and an EfficientNet-B0 stamp branch. "
            "Broker classifications are display-only and never model input."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.include_router(routes_alerts.router)
    app.include_router(routes_stamps.router)
    app.include_router(routes_method.router)
    app.include_router(routes_health.router)

    @app.get("/api/config")
    def config() -> dict:
        """What the frontend needs to know about how this instance is running."""
        return {
            "mode": settings.mode,
            "topics": list(settings.topics),
            "using_stubs": settings.use_stubs,
            "alerce_enabled": settings.alerce_enabled,
            "min_detections": settings.min_detections,
            "db_path": str(settings.db_path),
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "db_exists": settings.db_path.exists()}

    if WEB_DIR.exists():
        class RevalidatingStatic(StaticFiles):
            """Serve the dashboard assets with ``no-cache``.

            Not ``no-store``: the browser still caches and still gets cheap 304s
            via ETag. ``no-cache`` only forces *revalidation*, which is what
            stops an edited .js or .css being served stale from memory cache —
            the failure mode where the UI silently runs last week's code and you
            debug a bug that no longer exists.

            This is a single-user local demo, so the round trip costs nothing.
            """

            def is_not_modified(self, response_headers, request_headers) -> bool:
                return super().is_not_modified(response_headers, request_headers)

            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
                return response

        app.mount(
            "/static", RevalidatingStatic(directory=str(WEB_DIR)), name="static"
        )

        @app.get("/", include_in_schema=False)
        def index() -> HTMLResponse:
            """Serve the shell with cache-busted asset URLs.

            ``Cache-Control: no-cache`` is necessary but, in practice, not
            sufficient: some browsers still serve subresources from memory cache
            on a soft reload, so an edited .js keeps running the previous
            version and you debug a bug that is no longer in the source.

            Stamping each asset with a hash of its own mtime makes the URL
            change whenever the file does, which browsers cannot ignore. It is
            self-maintaining — no version constant to remember to bump.
            """
            html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

            def stamp(match: re.Match) -> str:
                url = match.group(1)
                asset = WEB_DIR / url[len("/static/") :]
                if not asset.exists():
                    return match.group(0)
                version = f"{int(asset.stat().st_mtime):x}"
                return match.group(0).replace(url, f"{url}?v={version}")

            html = re.sub(r'(?:src|href)="(/static/[^"?]+)"',
                          lambda m: stamp(m), html)
            return HTMLResponse(
                html, headers={"Cache-Control": "no-cache, must-revalidate"}
            )

    else:  # pragma: no cover - only if the web assets are missing
        @app.get("/", include_in_schema=False)
        def index_missing() -> JSONResponse:
            return JSONResponse(
                {"detail": f"dashboard assets not found at {WEB_DIR}"},
                status_code=503,
            )

    return app


app = create_app()
