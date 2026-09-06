# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""The FastAPI app, its routes, and the request pipeline that ties every
other router/ module together. handle_llm_request is the heart of it --
read its own docstring/body for the exact stage order (sanitize -> model
resolution -> thermal Stage 1 -> prefix-affinity Stage 2 -> routing
strategy -> record affinity -> forward). Nothing here implements routing
logic itself; it orchestrates calls into state/thermal/prefix_cache/
model_matching/sanitize/strategies/proxy, each of which owns one concern.
"""
import asyncio
import importlib
import json
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from router import (
    audit_log,
    metrics,
    model_matching,
    node_discovery,
    prefix_cache,
    proxy,
    rate_limit,
    sanitize,
    state,
    strategies,
    thermal,
)
from router.logging_setup import logger
from router.strategies import RoutingContext

# Load any self-hosted, out-of-tree strategy modules before the app starts
# serving -- each one's own @register_strategy(...) call does the rest.
# See router/strategies/__init__.py for the full "plug and play" story.
for _mod in os.getenv("LLM_ROUTER_EXTRA_STRATEGY_MODULES", "").split(","):
    if _mod.strip():
        importlib.import_module(_mod.strip())
        logger.info(f"Loaded extra routing strategy module: {_mod.strip()}")


async def update_models_cache_loop():
    """Background task to periodically refresh the active models list and thermal metrics from all nodes."""
    while True:
        try:
            state.reload_config_if_changed()

            tasks = [node_discovery.fetch_node_models(node) for node in state.CONFIG["nodes"]]
            temp_tasks = [thermal.fetch_node_temperature(node) for node in state.CONFIG["nodes"]]

            results = await asyncio.gather(*tasks)
            temp_results = await asyncio.gather(*temp_tasks)

            async with state.CACHE_LOCK:
                for node, models, temp in zip(state.CONFIG["nodes"], results, temp_results):
                    state.NODE_MODELS_CACHE[node["name"]] = models
                    if temp is not None:
                        state.NODE_TEMP_CACHE[node["name"]] = temp
                    elif node["name"] in state.NODE_TEMP_CACHE:
                        del state.NODE_TEMP_CACHE[node["name"]]

                    if models:
                        temp_str = f" ({temp:.1f}°C)" if temp is not None else ""
                        logger.debug(f"Node {node['name']}{temp_str} active models: {models}")
                    else:
                        logger.debug(f"Node {node['name']} is currently offline or has no models loaded.")
        except Exception as e:
            logger.error(f"Error in models cache update loop: {str(e)}")

        await asyncio.sleep(state.CONFIG["general_settings"].get("health_check_interval", 10))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown hook: loads config.yaml, initializes the
    shared httpx client (state.CLIENT -- every node/thermal HTTP call in
    this codebase reuses this one connection-pooled client), and starts
    the background node/thermal polling loop. Closes the client on
    shutdown."""
    state.CONFIG = state.load_config()
    state.ACTIVE_REQUESTS = {node["name"]: 0 for node in state.CONFIG.get("nodes", [])}
    state.CONFIG.setdefault("timeouts", {})
    state.CONFIG["timeouts"].setdefault("primary", 1.0)
    state.CONFIG["timeouts"].setdefault("fallback", 3.0)
    state.CONFIG["timeouts"].setdefault("request", 120.0)
    state.CONFIG.setdefault("general_settings", {})

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    state.CLIENT = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(None, connect=30.0))

    asyncio.create_task(update_models_cache_loop())

    yield

    await state.CLIENT.aclose()


app = FastAPI(
    title="LLM Cluster Router",
    version="1.0.0",
    description="A high-performance reverse proxy for routing LLM requests across a local GPU cluster.",
    lifespan=lifespan
)


@app.post("/_router/reload")
async def manual_reload_config():
    """Management endpoint to trigger instant hot-reload of config.yaml without container restarts."""
    reloaded = state.reload_config_if_changed()
    return {
        "status": "reloaded" if reloaded else "unchanged",
        "nodes_count": len(state.CONFIG.get("nodes", [])),
        "nodes": [n["name"] for n in state.CONFIG.get("nodes", [])],
        "routing_strategies_registered": strategies.registered_strategy_names(),
    }


@app.exception_handler(StarletteHTTPException)
async def openai_compatible_exception_handler(request: Request, exc: StarletteHTTPException):
    """Formats standard HTTPExceptions into OpenAI-compatible error payloads."""
    error_type = "invalid_request_error"
    if exc.status_code >= 500:
        error_type = "api_error"

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": error_type,
                "param": "model" if "model" in exc.detail.lower() else None,
                "code": str(exc.status_code)
            }
        }
    )


@app.get("/")
@app.get("/v1")
async def root_v1_info():
    """Returns cluster router information and OpenAPI base endpoints."""
    return {
        "status": "online",
        "service": "llm-cluster-router",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "completions": "/v1/completions",
            "health": "/health"
        }
    }


def _merge_model_entry(merged_models: dict, m) -> None:
    """Adds one node's reported model into the running merged_models dict
    that get_models() builds across all nodes, resolving a fallback
    context-window/max_model_len from the capabilities catalog if a node
    didn't report one, and keeping the MAXIMUM context length seen for a
    given model_id across every node currently serving it."""
    model_id = m.get("id") if isinstance(m, dict) else str(m)
    if not model_id:
        return

    model_obj = dict(m) if isinstance(m, dict) else {"id": model_id}
    model_obj.setdefault("object", "model")
    model_obj.setdefault("created", 1677610602)
    model_obj.setdefault("owned_by", "llm-cluster")

    if "max_model_len" not in model_obj and "context_window" not in model_obj:
        caps = model_matching.get_model_capabilities(model_id)
        ctx_len = caps.get("context_window", 131072)
        model_obj["max_model_len"] = ctx_len
        model_obj["context_window"] = ctx_len
    elif "max_model_len" in model_obj and "context_window" not in model_obj:
        model_obj["context_window"] = model_obj["max_model_len"]
    elif "context_window" in model_obj and "max_model_len" not in model_obj:
        model_obj["max_model_len"] = model_obj["context_window"]

    if model_id not in merged_models:
        merged_models[model_id] = model_obj
    else:
        # Take the MAXIMUM context window supported across all nodes serving this model
        existing_len = merged_models[model_id].get("max_model_len", 0)
        new_len = model_obj.get("max_model_len", 0)
        max_len = max(existing_len, new_len)
        merged_models[model_id]["max_model_len"] = max_len
        merged_models[model_id]["context_window"] = max_len


@app.get("/v1/models")
async def get_models():
    """Returns the unified list of currently loaded models across all active nodes,
    aggregating the maximum context_window / max_model_len supported across nodes."""
    merged_models = {}

    async with state.CACHE_LOCK:
        for node_name, models in state.NODE_MODELS_CACHE.items():
            for m in models:
                _merge_model_entry(merged_models, m)

    # Fallback: if cache is empty, query once in real-time
    if not merged_models:
        logger.info("Models cache empty; performing real-time query on all nodes...")
        tasks = [node_discovery.fetch_node_models(node) for node in state.CONFIG["nodes"]]
        results = await asyncio.gather(*tasks)
        for models in results:
            for m in models:
                _merge_model_entry(merged_models, m)

    return {"object": "list", "data": list(merged_models.values())}


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
@app.post("/v1/embeddings")
async def handle_llm_request(request: Request):
    """Parses model request and routes to a node currently serving that model."""
    path = request.url.path.lstrip('/')
    if path.startswith("v1/"):
        path = path[3:]
    method = request.method
    headers = dict(request.headers)
    body = await request.body()

    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    auth_user = audit_log.parse_auth_user(headers)
    prompt = audit_log.parse_request_prompt(body)
    trace_id = audit_log.parse_trace_id(headers)

    client_id = auth_user if auth_user != "anonymous" else client_ip
    if not rate_limit.check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded for client '{client_id}'. Max allowed requests per minute reached.")

    try:
        json_data = await request.json()
        requested_model = json_data.get("model")
        is_stream = json_data.get("stream", False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    prefix_hash = prefix_cache.get_prefix_hash(json_data)
    cached_prefix_node = await prefix_cache.lookup(prefix_hash)

    if not requested_model:
        raise HTTPException(status_code=400, detail="Missing 'model' parameter in request body")

    original_model = requested_model
    body_to_send = body
    modified = False

    if "tools" in json_data or "functions" in json_data:
        try:
            if sanitize.sanitize_tools(json_data):
                modified = True
        except Exception as e:
            logger.warning(f"Error sanitizing tools/functions parameter: {str(e)}")

    messages = json_data.get("messages")
    if isinstance(messages, list):
        try:
            if sanitize.sanitize_messages(messages):
                modified = True
        except Exception as e:
            logger.warning(f"Error sanitizing messages history: {str(e)}")

    try:
        if model_matching.check_and_reroute_capabilities(json_data):
            modified = True
            requested_model = json_data["model"]
    except Exception as e:
        logger.warning(f"Error performing capabilities fallback checks: {str(e)}")

    if modified:
        body_to_send = json.dumps(json_data).encode("utf-8")

    eligible_nodes = await _resolve_eligible_nodes(requested_model, json_data)
    if eligible_nodes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{original_model}' is not currently loaded on any online cluster nodes, and no active fallback models are currently served by the cluster."
        )
    eligible_nodes, requested_model, body_to_send = eligible_nodes

    routing_config = state.CONFIG.get("routing", {})
    routing_mode = routing_config.get("mode", "smart").lower()

    # STAGE 1: critical thermal hard-block -- runs regardless of strategy.
    eligible_nodes = thermal.apply_stage1_critical_filter(eligible_nodes, requested_model)

    # STAGE 2: prefix-cache affinity (with thermal-cooling override) -- also
    # runs regardless of strategy, since it's about correctness/performance
    # of caching, not "which strategy is configured."
    selected_node = thermal.apply_stage2_prefix_affinity(cached_prefix_node, eligible_nodes)
    if selected_node is not None:
        if metrics.HAS_PROMETHEUS:
            metrics.ROUTER_PREFIX_AFFINITY_ROUTED.labels(node=selected_node['name']).inc()
        logger.info(f"Prefix KV-cache hit for hash '{prefix_hash}'. Routing to warm cache node '{selected_node['name']}'")
    else:
        strategy = strategies.get_strategy(routing_mode)
        selected_node = strategy(RoutingContext(
            eligible_nodes=eligible_nodes,
            requested_model=requested_model,
            prompt=prompt,
            json_data=json_data,
            headers=headers,
            auth_user=auth_user,
            client_ip=client_ip,
            routing_config=routing_config,
        ))

    # Record prefix cache location for future requests -- unconditional,
    # single call site, regardless of which of the two branches above
    # produced selected_node. See router/prefix_cache.py's docstring for
    # why this used to only work for one specific strategy.
    if prefix_hash and selected_node:
        await prefix_cache.record(prefix_hash, selected_node["name"])

    return await proxy.forward_request(
        node=selected_node,
        path=path,
        method=method,
        headers=headers,
        content=body_to_send,
        client_ip=client_ip,
        auth_user=auth_user,
        requested_model=requested_model,
        prompt=prompt,
        is_stream=is_stream,
        original_model=original_model,
        trace_id=trace_id
    )


async def _resolve_eligible_nodes(requested_model: str, json_data: dict):
    """Finds nodes serving requested_model, falling back to a real-time
    query and then to an auto-rewritten active model if nothing matches.
    Returns (eligible_nodes, requested_model, body_to_send) or None if no
    node in the cluster can serve anything at all."""
    body_to_send = json.dumps(json_data).encode("utf-8")
    eligible_nodes = []
    async with state.CACHE_LOCK:
        for node in state.CONFIG["nodes"]:
            active_models = state.NODE_MODELS_CACHE.get(node["name"], [])
            active_ids = [m.get("id") if isinstance(m, dict) else str(m) for m in active_models]
            for a_id in active_ids:
                if model_matching.is_model_matching(requested_model, a_id):
                    eligible_nodes.append(node)
                    if requested_model != a_id:
                        json_data["model"] = a_id
                        body_to_send = json.dumps(json_data).encode("utf-8")
                    break

    if not eligible_nodes:
        logger.info(f"Model '{requested_model}' not found in cache. Querying nodes in real-time...")
        for node in state.CONFIG["nodes"]:
            models = await node_discovery.fetch_node_models(node)
            active_ids = [m.get("id") if isinstance(m, dict) else str(m) for m in models]
            if requested_model in active_ids:
                eligible_nodes.append(node)
                async with state.CACHE_LOCK:
                    state.NODE_MODELS_CACHE[node["name"]] = models

    if not eligible_nodes:
        all_active_models = []
        async with state.CACHE_LOCK:
            for models in state.NODE_MODELS_CACHE.values():
                for m in models:
                    m_id = m.get("id") if isinstance(m, dict) else str(m)
                    if m_id and m_id not in all_active_models:
                        all_active_models.append(m_id)

        if all_active_models:
            fallback_model = all_active_models[0]
            logger.info(f"Model '{requested_model}' not found in cluster. Auto-rewriting model parameter to active fallback: '{fallback_model}'")
            try:
                json_data["model"] = fallback_model
                body_to_send = json.dumps(json_data).encode("utf-8")
                requested_model = fallback_model
            except Exception as e:
                logger.error(f"Failed to rewrite request body: {str(e)}")

            async with state.CACHE_LOCK:
                for node in state.CONFIG["nodes"]:
                    active_models = state.NODE_MODELS_CACHE.get(node["name"], [])
                    active_ids = [m.get("id") if isinstance(m, dict) else str(m) for m in active_models]
                    if requested_model in active_ids:
                        eligible_nodes.append(node)

    if not eligible_nodes:
        return None

    return eligible_nodes, requested_model, body_to_send


@app.get("/health")
async def health():
    """Liveness check for the router proxy itself."""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus scrape endpoint -- see router/metrics.py and the
    README's "Metrics (Prometheus)" section for what each counter means."""
    body, media_type = metrics.metrics_response_body_and_type()
    return Response(content=body, media_type=media_type)
