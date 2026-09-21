"""Bound upload ingress before Starlette starts spooling multipart files.

The byte budget includes multipart headers and boundaries, not just file payloads.
Admission is process-local and released when the response starts (or on failure).
"""

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _UploadTooLarge(MultiPartException):
    """Let Starlette close partial multipart spools on the rejection path."""


class _UploadDisconnected(MultiPartException):
    """Ensure partial multipart files close when a client goes away."""


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int, max_concurrent: int) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.max_concurrent = max_concurrent
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("path") != "/api/analyze"
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(
                scope, receive, send
            )
            return
        if self.max_bytes > 0 and declared > self.max_bytes:
            await JSONResponse(
                {"detail": "Upload exceeds the configured byte limit"}, status_code=413
            )(scope, receive, send)
            return
        if self.active >= self.max_concurrent:
            await JSONResponse(
                {"detail": "Upload capacity exhausted; retry later"}, status_code=503
            )(scope, receive, send)
            return
        self.active += 1
        admitted = True
        exceeded = False
        total = 0
        responded = False

        def release() -> None:
            nonlocal admitted
            if admitted:
                self.active -= 1
                admitted = False

        async def bounded_receive() -> Message:
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.disconnect":
                raise _UploadDisconnected("Upload disconnected")
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if self.max_bytes > 0 and total > self.max_bytes:
                    exceeded = True
                    raise _UploadTooLarge("Upload exceeds the configured byte limit")
            return message

        async def bounded_send(message: Message) -> None:
            nonlocal responded
            if message["type"] == "http.response.start":
                release()
                responded = True
                # Request.form translates MultiPartException to HTTP 400 after
                # closing its temporary files. Preserve its body, correct status.
                if exceeded:
                    message = {**message, "status": 413}
            await send(message)

        try:
            await self.app(scope, bounded_receive, bounded_send)
        except _UploadDisconnected:
            pass
        except _UploadTooLarge:
            if not responded:
                await JSONResponse(
                    {"detail": "Upload exceeds the configured byte limit"}, status_code=413
                )(scope, receive, send)
        finally:
            release()
