"""RFC 9457 problem+json error handling for imprint-server."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class ImprintError(Exception):
    """Raised by route handlers to produce a problem+json response."""

    def __init__(self, *, status: int, title: str, detail: str) -> None:
        self.status = status
        self.title = title
        self.detail = detail
        super().__init__(detail)


async def imprint_error_handler(request: Request, exc: ImprintError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={
            "type": "about:blank",
            "title": exc.title,
            "status": exc.status,
            "detail": exc.detail,
        },
        headers={"Content-Type": "application/problem+json"},
    )


# -- Convenience constructors -------------------------------------------------


def not_found(detail: str) -> ImprintError:
    return ImprintError(status=404, title="Not Found", detail=detail)


def bad_request(detail: str) -> ImprintError:
    return ImprintError(status=422, title="Unprocessable Request", detail=detail)


def internal_error(detail: str) -> ImprintError:
    return ImprintError(status=500, title="Internal Server Error", detail=detail)
