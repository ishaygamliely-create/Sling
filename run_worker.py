"""
run_worker.py — local development launcher for the Sling Worker.

Adds worker/ to sys.path so that bare imports (redis_store, processor)
resolve correctly when launched from the repo root.

Usage (from repo root):
    $env:DEV_MODE="1"; $env:WORKER_AUTH_TOKEN="localtest"
    python run_worker.py

This file is for LOCAL development only.
The Fly.io Docker image launches uvicorn directly from /app where all
worker files are copied flat, so this file is not needed in production.
"""

import sys
import os

# Add worker/ to path so bare imports like `import redis_store` resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "worker"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=9000, reload=True,
                reload_dirs=["worker"])
