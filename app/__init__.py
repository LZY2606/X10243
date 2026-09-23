"""Local ITC review application package.

``uvicorn app:app`` resolves the FastAPI instance lazily so importing a single
submodule (e.g. ``app.binding`` from scripts/tests) never starts a server.
"""

def __getattr__(name):
    if name == "app":
        from .server import app
        return app
    raise AttributeError(name)
