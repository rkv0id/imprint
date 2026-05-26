# ruff: noqa: E501
"""Admin dashboard served at /admin.

Loads the HTML template from api/templates/dashboard.html via
importlib.resources and injects the logo SVG. The template contains
all CSS and JS inline, making the page fully self-contained.

Auth: /admin is exempt from AuthMiddleware so the HTML loads without a
Bearer token. The JS inside handles auth for all subsequent API calls.
"""

from __future__ import annotations

import importlib.resources as _ir

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# The actual logo SVG from docs/media/mark-light.svg.
# Inlined here so the template has no filesystem dependency.
_LOGO_SVG = (
    '<svg width="32" height="32" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M 12 6 H 52 A 6 6 0 0 1 58 12 V 30 H 6 V 12 A 6 6 0 0 1 12 6 Z" fill="#0d9488"/>'
    '<rect x="38" y="10" width="6" height="14" fill="#f5f5f0"/>'
    '<path d="M 6 34 H 58 V 52 A 6 6 0 0 1 52 58 H 12 A 6 6 0 0 1 6 52 Z" fill="none" stroke="#0c0f1a" stroke-width="2"/>'
    '<circle cx="13" cy="42" r="2" fill="#0d9488"/>'
    '<rect x="18" y="40" width="13" height="4" rx="1" fill="#0c0f1a"/>'
    '<line x1="32" y1="42" x2="36" y2="42" stroke="#0c0f1a" stroke-width="1.4"/>'
    '<rect x="37" y="40" width="13" height="4" rx="1" fill="#0c0f1a"/>'
    '<circle cx="13" cy="50" r="2" fill="none" stroke="#0c0f1a" stroke-width="1" opacity=".5"/>'
    '<rect x="18" y="48" width="13" height="4" rx="1" fill="#0c0f1a" opacity=".4"/>'
    '<line x1="32" y1="50" x2="36" y2="50" stroke="#0c0f1a" stroke-width="1.4" opacity=".4"/>'
    '<rect x="37" y="48" width="13" height="4" rx="1" fill="#0c0f1a" opacity=".4"/>'
    "</svg>"
)

_TEMPLATE = (_ir.files("imprint_server.api") / "templates" / "dashboard.html").read_text(
    encoding="utf-8"
)
_DASHBOARD_HTML = _TEMPLATE.replace("{{LOGO_SVG}}", _LOGO_SVG)


@router.get(
    "/admin",
    operation_id="admin_dashboard",
    tags=["system"],
    summary="Read-only admin dashboard",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def admin_dashboard() -> HTMLResponse:
    """Serve the read-only admin dashboard.

    Self-contained HTML/CSS/JS page. All data is fetched client-side
    from the existing REST API using the operator's Bearer token.
    The page itself is auth-exempt; API calls inside it are not.
    """
    return HTMLResponse(_DASHBOARD_HTML)
