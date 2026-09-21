"""Upload admission applies to actual ASGI bytes before multipart parsing."""

import asyncio

import pytest

from scalar_forensic.web.ingress import UploadLimitMiddleware


def test_declared_oversize_never_calls_parser():
    async def run():
        called = False
        sent = []

        async def app(scope, receive, send):
            nonlocal called
            called = True

        async def send(message):
            sent.append(message)

        middleware = UploadLimitMiddleware(app, max_bytes=10, max_concurrent=1)
        await middleware(
            {
                "type": "http",
                "path": "/api/analyze",
                "method": "POST",
                "headers": [(b"content-length", b"11")],
            },
            None,
            send,
        )
        assert not called
        assert sent[0]["status"] == 413

    asyncio.run(run())


def test_chunked_oversize_stops_receive_and_frees_admission():
    async def run():
        received = 0
        sent = []

        async def receive():
            nonlocal received
            received += 1
            return {"type": "http.request", "body": b"123456", "more_body": True}

        async def app(scope, receive, send):
            while True:
                await receive()

        async def send(message):
            sent.append(message)

        middleware = UploadLimitMiddleware(app, max_bytes=10, max_concurrent=1)
        await middleware(
            {"type": "http", "path": "/api/analyze", "method": "POST", "headers": []}, receive, send
        )
        assert received == 2
        assert sent[0]["status"] == 413
        assert middleware.active == 0

    asyncio.run(run())


def test_concurrent_upload_refused_before_receive():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        sent = []

        async def app(scope, receive, send):
            entered.set()
            await release.wait()

        async def send(message):
            sent.append(message)

        middleware = UploadLimitMiddleware(app, max_bytes=10, max_concurrent=1)
        scope = {"type": "http", "path": "/api/analyze", "method": "POST", "headers": []}
        first = asyncio.create_task(middleware(scope, None, send))
        await entered.wait()
        await middleware(scope, None, send)
        assert sent[0]["status"] == 503
        release.set()
        await first
        assert middleware.active == 0

    asyncio.run(run())


@pytest.mark.parametrize("disconnect", [False, True])
def test_multipart_limit_closes_partial_spool(monkeypatch, disconnect):
    import starlette.formparsers as parsers
    from fastapi import FastAPI, UploadFile

    spools = []
    original = parsers.SpooledTemporaryFile

    def spool(*args, **kwargs):
        result = original(*args, **kwargs)
        spools.append(result)
        return result

    monkeypatch.setattr(parsers, "SpooledTemporaryFile", spool)

    async def run():
        called = False
        received = 0
        sent = []
        app = FastAPI()

        @app.post("/api/analyze")
        async def endpoint(files: list[UploadFile]):
            nonlocal called
            called = True
            return {}

        chunks = iter(
            [
                b'--test\r\nContent-Disposition: form-data; name="files"; '
                b'filename="a.bin"\r\n\r\n1234',
                b"x" * 300,
                b"x" * 300,
            ]
        )

        async def receive():
            nonlocal received
            received += 1
            if disconnect and received == 2:
                return {"type": "http.disconnect"}
            return {"type": "http.request", "body": next(chunks), "more_body": True}

        async def send(message):
            sent.append(message)

        middleware = UploadLimitMiddleware(app, max_bytes=200, max_concurrent=1)
        await middleware(
            {
                "type": "http",
                "path": "/api/analyze",
                "method": "POST",
                "headers": [(b"content-type", b"multipart/form-data; boundary=test")],
                "query_string": b"",
            },
            receive,
            send,
        )
        assert not called
        assert received == 2
        assert sent[0]["status"] == (400 if disconnect else 413)
        assert spools and all(spool.closed for spool in spools)
        assert middleware.active == 0

    asyncio.run(run())


def test_disconnect_releases_admission():
    async def run():
        async def receive():
            return {"type": "http.disconnect"}

        async def app(scope, receive, send):
            await receive()

        middleware = UploadLimitMiddleware(app, max_bytes=10, max_concurrent=1)
        await middleware(
            {"type": "http", "path": "/api/analyze", "method": "POST", "headers": []}, receive, None
        )
        assert middleware.active == 0

    asyncio.run(run())


def test_normal_multipart_upload_reaches_endpoint():
    from fastapi import FastAPI, UploadFile

    async def run():
        app = FastAPI()
        received = []
        sent = []

        @app.post("/api/analyze")
        async def endpoint(files: list[UploadFile]):
            received.extend([await file.read() for file in files])
            return {"count": len(files)}

        body = (
            b'--test\r\nContent-Disposition: form-data; name="files"; filename="a.bin"\r\n\r\n'
            b'first\r\n--test\r\nContent-Disposition: form-data; name="files"; '
            b'filename="b.bin"\r\n\r\n'
            b"second\r\n--test--\r\n"
        )
        chunks = iter([body[:110], body[110:]])
        count = 0

        async def receive():
            nonlocal count
            count += 1
            return {"type": "http.request", "body": next(chunks), "more_body": count == 1}

        async def send(message):
            sent.append(message)

        middleware = UploadLimitMiddleware(app, max_bytes=1024, max_concurrent=1)
        await middleware(
            {
                "type": "http",
                "path": "/api/analyze",
                "method": "POST",
                "headers": [(b"content-type", b"multipart/form-data; boundary=test")],
                "query_string": b"",
            },
            receive,
            send,
        )
        assert sent[0]["status"] == 200
        assert received == [b"first", b"second"]
        assert middleware.active == 0

    asyncio.run(run())
