"""In-browser video playback (spec: docs/specs/video-playback-transcode.md).

A self-contained subsystem following the precedent of ``scalar_forensic.faces``:
everything that decides how a source video reaches the player — codec and
container classification, the lossless rewrap, the source digest, the bounded
viewing-copy cache and the routes that serve them — lives here.

Public API is the router; everything else is internal to the package.
"""

__all__ = ["router"]


def __getattr__(name):
    # Decoder/encoder workers import package helpers without loading web routes
    # or inference dependencies. Preserve the public router import for the app.
    if name == "router":
        from scalar_forensic.video_playback.routes import router

        globals()[name] = router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
