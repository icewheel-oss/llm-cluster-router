# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Entrypoint kept at the repo root so `uvicorn main:app` (Dockerfile,
docker-compose.yml, and any existing self-hosted deployment's own
tooling) keeps working unchanged. All actual logic lives in router/ --
see router/app.py for the FastAPI app and request pipeline,
router/strategies/ for the pluggable routing-strategy registry."""
from router.app import app  # noqa: F401
