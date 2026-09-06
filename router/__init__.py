"""The llm-cluster-router package.

See router/app.py for the FastAPI app and the request pipeline
(handle_llm_request), and router/strategies/ for the pluggable
routing-strategy registry. main.py at the repo root is just
`from router.app import app`, kept there so `uvicorn main:app` keeps
working for every existing deployment.
"""
