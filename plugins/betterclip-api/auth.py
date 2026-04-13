"""
BetterClip API — Token authentication middleware
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

HEADER_NAME = "X-BetterClip-Token"


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Reject any request missing or carrying an invalid token."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        # Allow CORS preflight through
        if request.method == "OPTIONS":
            return await call_next(request)

        provided = request.headers.get(HEADER_NAME)
        if not provided or provided != self.token:
            return JSONResponse(
                {"error": "unauthorized", "detail": "Missing or invalid X-BetterClip-Token"},
                status_code=401,
            )
        return await call_next(request)
