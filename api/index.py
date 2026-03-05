# api/index.py — Vercel entrypoint
# Vercel looks for the ASGI `app` variable in this file.
from server import app  # noqa: F401  (re-exported for Vercel)
