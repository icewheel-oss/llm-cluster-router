# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import asyncio
import logging
import random
import os
import yaml
import json
import base64
import zlib
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, List, Any

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm-cluster-router")

# Global state
CONFIG: Dict[str, Any] = {}
CLIENT: httpx.AsyncClient = None
LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", "5044"))
# Cache of active models per node: {node_name: [model_id_1, model_id_2, ...]}
NODE_MODELS_CACHE: Dict[str, List[str]] = {}
ACTIVE_REQUESTS: Dict[str, int] = {}
CACHE_LOCK = asyncio.Lock()


def increment_active_requests(node_name: str):
    ACTIVE_REQUESTS[node_name] = ACTIVE_REQUESTS.get(node_name, 0) + 1


def decrement_active_requests(node_name: str):
    ACTIVE_REQUESTS[node_name] = max(0, ACTIVE_REQUESTS.get(node_name, 0) - 1)


def load_config() -> Dict[str, Any]:
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def fetch_node_models(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Queries a node's /models endpoint, preserving full model objects (including max_model_len/context_window)
    and passing custom node headers if configured."""
    primary_url = node["primary"]
    backup = node.get("backup")
    node_headers = node.get("headers", {})
    req_headers = {**node_headers}
    
    # Try Primary
    try:
        url = f"{primary_url.rstrip('/')}/models"
        logger.debug(f"Querying models from primary node {node['name']} at {url}")
        resp = await CLIENT.get(url, headers=req_headers, timeout=CONFIG["timeouts"]["primary"])
        if resp.status_code == 200:
            data = resp.json()
            models_data = data.get("data", [])
            # Return full model dictionaries or wrap string list items into dicts
            result = []
            for item in models_data:
                if isinstance(item, dict):
                    result.append(item)
                elif isinstance(item, str):
                    result.append({"id": item})
            return result
    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning(f"Primary endpoint failed for {node['name']} ({str(e)}). Trying backup...")
        
    # Try Backup
    if backup:
        try:
            url = f"{backup.rstrip('/')}/models"
            logger.debug(f"Querying models from backup node {node['name']} at {url}")
            resp = await CLIENT.get(url, headers=req_headers, timeout=CONFIG["timeouts"]["fallback"])
            if resp.status_code == 200:
                data = resp.json()
                models_data = data.get("data", [])
                result = []
                for item in models_data:
                    if isinstance(item, dict):
                        result.append(item)
                    elif isinstance(item, str):
                        result.append({"id": item})
                return result
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.error(f"Backup endpoint also failed for {node['name']} ({str(e)})")
            
    return []


# Cache of active models per node: {node_name: [model_obj_1, model_obj_2, ...]}
NODE_MODELS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
NODE_TEMP_CACHE: Dict[str, float] = {}

# Prefix KV-Cache tracking: {prefix_hash: (node_name, timestamp)}
PREFIX_CACHE: Dict[str, tuple] = {}

# Rate limiting tracking: {client_identifier: [timestamp_1, timestamp_2, ...]}
RATE_LIMIT_CACHE: Dict[str, List[float]] = {}

CACHE_LOCK = asyncio.Lock()


def get_prefix_hash(json_data: dict) -> str:
    """Extracts system prompt or initial prompt prefix and returns a CRC32 hash string.
    Returns empty string if prefix length is below min_prefix_length.
    """
    prefix_cfg = CONFIG.get("prefix_cache_routing", {})
    if not prefix_cfg.get("enabled", True):
        return ""

    min_len = prefix_cfg.get("min_prefix_length", 50)
    messages = json_data.get("messages")
    prefix_text = ""

    if isinstance(messages, list) and messages:
        # Check system message first
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                prefix_text = str(msg.get("content", ""))
                break
        # Fall back to first user message if no system message
        if not prefix_text and isinstance(messages[0], dict):
            prefix_text = str(messages[0].get("content", ""))
    elif "prompt" in json_data:
        prefix_text = str(json_data.get("prompt", ""))

    if len(prefix_text) >= min_len:
        return f"{zlib.crc32(prefix_text.encode('utf-8')):x}"

    return ""


def check_rate_limit(client_id: str) -> bool:
    """Checks if client exceeds configured requests per minute limit.
    Returns True if request is ALLOWED, False if RATE LIMITED.
    """
    rl_cfg = CONFIG.get("rate_limiting", {})
    if not rl_cfg.get("enabled", False):
        return True

    limit = rl_cfg.get("requests_per_minute", 60)
    now = time.time()

    asyncio.run_coroutine_threadsafe

    history = RATE_LIMIT_CACHE.get(client_id, [])
    # Keep timestamps within the last 60 seconds
    history = [t for t in history if now - t < 60.0]
    
    if len(history) >= limit:
        return False

    history.append(now)
    RATE_LIMIT_CACHE[client_id] = history
    return True



async def fetch_node_temperature(node: Dict[str, str]) -> float:
    """Attempts to fetch node hardware/GPU temperature via node-exporter on port 9100.
    Returns maximum temperature in Celsius or None if telemetry/exporter is unavailable.
    """
    thermal_cfg = CONFIG.get("thermal_routing", {})
    if not thermal_cfg.get("enabled", True):
        return None

    # Derive host/IP from primary node URL
    primary_url = node.get("primary", "")
    try:
        host = primary_url.split("://")[-1].split(":")[0]
    except Exception:
        return None

    port = thermal_cfg.get("exporter_port", 9100)
    metrics_url = f"http://{host}:{port}/metrics"

    try:
        resp = await CLIENT.get(metrics_url, timeout=1.5)
        if resp.status_code == 200:
            temps = []
            for line in resp.text.splitlines():
                if line.startswith("node_hwmon_temp_celsius{") or line.startswith("node_thermal_zone_temp{"):
                    try:
                        val = float(line.split()[-1])
                        # Filter out invalid sensor sentinel values (e.g. >150 or <=0)
                        if 0 < val < 150:
                            temps.append(val)
                    except ValueError:
                        continue
            if temps:
                return max(temps)
    except Exception:
        pass
    return None


async def update_models_cache_loop():
    """Background task to periodically refresh the active models list and thermal metrics from all nodes."""
    while True:
        try:
            tasks = [fetch_node_models(node) for node in CONFIG["nodes"]]
            temp_tasks = [fetch_node_temperature(node) for node in CONFIG["nodes"]]
            
            results = await asyncio.gather(*tasks)
            temp_results = await asyncio.gather(*temp_tasks)
            
            async with CACHE_LOCK:
                for node, models, temp in zip(CONFIG["nodes"], results, temp_results):
                    NODE_MODELS_CACHE[node["name"]] = models
                    if temp is not None:
                        NODE_TEMP_CACHE[node["name"]] = temp
                    elif node["name"] in NODE_TEMP_CACHE:
                        del NODE_TEMP_CACHE[node["name"]]

                    if models:
                        temp_str = f" ({temp:.1f}°C)" if temp is not None else ""
                        logger.debug(f"Node {node['name']}{temp_str} active models: {models}")
                    else:
                        logger.debug(f"Node {node['name']} is currently offline or has no models loaded.")
        except Exception as e:
            logger.error(f"Error in models cache update loop: {str(e)}")
            
        await asyncio.sleep(CONFIG["general_settings"].get("health_check_interval", 10))



@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONFIG, CLIENT, ACTIVE_REQUESTS
    # Load configuration
    CONFIG = load_config()
    ACTIVE_REQUESTS = {node["name"]: 0 for node in CONFIG.get("nodes", [])}
    # Normalize settings
    CONFIG.setdefault("timeouts", {})
    CONFIG["timeouts"].setdefault("primary", 1.0)
    CONFIG["timeouts"].setdefault("fallback", 3.0)
    CONFIG["timeouts"].setdefault("request", 120.0)
    CONFIG.setdefault("general_settings", {})
    
    # Initialize global async HTTP client
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    CLIENT = httpx.AsyncClient(limits=limits)
    
    # Start background cache sync
    asyncio.create_task(update_models_cache_loop())
    
    yield
    
    # Shutdown client
    await CLIENT.aclose()

app = FastAPI(
    title="LLM Cluster Router",
    version="1.0.0",
    description="A high-performance reverse proxy for routing LLM requests across a local GPU cluster.",
    lifespan=lifespan
)


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


def parse_auth_user(headers: Dict[str, str]) -> str:
    auth = headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            cred = base64.b64decode(auth[6:]).decode("utf-8")
            return cred.split(":")[0]
        except Exception:
            pass
    return "anonymous"


def parse_request_prompt(content: bytes) -> str:
    try:
        data = json.loads(content)
        messages = data.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    return msg.get("content", "")
    except Exception:
        pass
    return ""


def parse_trace_id(headers: Dict[str, str]) -> str:
    # Check traceparent (W3C standard format: version-trace_id-parent_id-trace_flags)
    traceparent = headers.get("traceparent")
    if not traceparent:
        for k, v in headers.items():
            if k.lower() == "traceparent":
                traceparent = v
                break
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 2:
            return parts[1]
            
    # Check X-Request-ID
    x_req_id = headers.get("x-request-id")
    if not x_req_id:
        for k, v in headers.items():
            if k.lower() == "x-request-id":
                x_req_id = v
                break
    if x_req_id:
        return x_req_id
        
    return None


async def send_log_to_logstash(log_entry: dict):
    if not LOGSTASH_HOST:
        return
    log_entry["app_name"] = "llm-cluster-router"
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(LOGSTASH_HOST, LOGSTASH_PORT),
            timeout=2.0
        )
        writer.write(json.dumps(log_entry).encode("utf-8") + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        logger.warning(f"Failed to ship audit log to Logstash: {str(e)}")


async def stream_and_log(
    resp: httpx.Response,
    node_name: str,
    method: str,
    path: str,
    client_ip: str,
    auth_user: str,
    requested_model: str,
    prompt: str,
    start_time: float,
    is_stream: bool,
    original_model: str = None,
    trace_id: str = None
):
    full_response_bytes = []
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
            full_response_bytes.append(chunk)
    except asyncio.CancelledError:
        logger.warning(f"Client disconnected/cancelled during generation for model '{requested_model}' on node '{node_name}'")
        raise
    finally:
        try:
            aclose_func = getattr(resp, "aclose", None)
            if aclose_func:
                res = aclose_func()
                if asyncio.iscoroutine(res):
                    await res
        except Exception as e:
            logger.warning(f"Error closing response connection: {str(e)}")
            
        decrement_active_requests(node_name)
        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)
        response_data = b"".join(full_response_bytes)
        
        response_text = ""
        completion_id = None
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        if not is_stream:
            try:
                data = json.loads(response_data)
                completion_id = data.get("id")
                choices = data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    total_tokens = usage.get("total_tokens")
            except Exception:
                pass
        else:
            try:
                lines = response_data.decode("utf-8", errors="ignore").split("\n")
                chunks_text = []
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            continue
                        try:
                            chunk_data = json.loads(data_str)
                            if not completion_id:
                                completion_id = chunk_data.get("id")
                            choices = chunk_data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "content" in delta:
                                    chunks_text.append(delta["content"])
                            usage = chunk_data.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens")
                                completion_tokens = usage.get("completion_tokens")
                                total_tokens = usage.get("total_tokens")
                        except Exception:
                            pass
                response_text = "".join(chunks_text)
            except Exception:
                pass
                
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "client_ip": client_ip,
            "auth_user": auth_user,
            "model": requested_model,
            "original_model": original_model or requested_model,
            "rewritten": original_model is not None and original_model != requested_model,
            "trace_id": trace_id,
            "completion_id": completion_id,
            "method": method,
            "path": path,
            "status_code": resp.status_code,
            "latency_ms": latency_ms,
            "node": node_name,
            "prompt": prompt,
            "response": response_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stream": is_stream
        }
        logger.info(f"AUDIT_LOG: {json.dumps(log_entry)}")
        asyncio.create_task(send_log_to_logstash(log_entry))


async def forward_request(
    node: Dict[str, str],
    path: str,
    method: str,
    headers: Dict[str, str],
    content: bytes,
    client_ip: str,
    auth_user: str,
    requested_model: str,
    prompt: str,
    is_stream: bool,
    original_model: str = None,
    trace_id: str = None
) -> StreamingResponse:
    """Forwards request to a node, routing via primary Ethernet or failing over to backup WiFi."""
    primary_base = node["primary"].rstrip('/')
    backup_base = node.get("backup", "").rstrip('/')
    
    headers_to_send = {k: v for k, v in headers.items() if k.lower() not in ("host", "content-length")}
    node_headers = node.get("headers", {})
    if isinstance(node_headers, dict):
        headers_to_send.update(node_headers)
    start_time = time.time()
    
    increment_active_requests(node["name"])
    success = False
    try:
        # Try Primary
        try:
            url = f"{primary_base}/{path}"
            logger.info(f"Forwarding {method} request to primary node {node['name']} at {url}")
            req = CLIENT.build_request(method, url, headers=headers_to_send, content=content, timeout=CONFIG["timeouts"]["request"])
            resp = await CLIENT.send(req, stream=True)
            success = True
            return StreamingResponse(
                stream_and_log(
                    resp=resp,
                    node_name=node["name"],
                    method=method,
                    path=path,
                    client_ip=client_ip,
                    auth_user=auth_user,
                    requested_model=requested_model,
                    prompt=prompt,
                    start_time=start_time,
                    is_stream=is_stream,
                    original_model=original_model,
                    trace_id=trace_id
                ),
                status_code=resp.status_code,
                headers=dict(resp.headers)
            )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.warning(f"Failed to forward to primary node {node['name']} ({str(e)}). Attempting backup failover...")

        # Try Backup
        if backup_base:
            try:
                url = f"{backup_base}/{path}"
                logger.info(f"Forwarding {method} request to backup node {node['name']} at {url}")
                req = CLIENT.build_request(method, url, headers=headers_to_send, content=content, timeout=CONFIG["timeouts"]["request"])
                resp = await CLIENT.send(req, stream=True)
                success = True
                return StreamingResponse(
                    stream_and_log(
                        resp=resp,
                        node_name=node["name"],
                        method=method,
                        path=path,
                        client_ip=client_ip,
                        auth_user=auth_user,
                        requested_model=requested_model,
                        prompt=prompt,
                        start_time=start_time,
                        is_stream=is_stream,
                        original_model=original_model,
                        trace_id=trace_id
                    ),
                    status_code=resp.status_code,
                    headers=dict(resp.headers)
                )
            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.error(f"Backup endpoint failed too for {node['name']} ({str(e)})")
                raise HTTPException(status_code=502, detail=f"Both primary and backup endpoints failed for node {node['name']}")
                
        raise HTTPException(status_code=502, detail=f"Primary endpoint failed for node {node['name']} and no backup is configured.")
    finally:
        if not success:
            decrement_active_requests(node["name"])


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


@app.get("/v1/models")
async def get_models():
    """Returns the unified list of currently loaded models across all active nodes,
    aggregating the maximum context_window / max_model_len supported across nodes."""
    merged_models = {}
    
    async with CACHE_LOCK:
        for node_name, models in NODE_MODELS_CACHE.items():
            for m in models:
                model_id = m.get("id") if isinstance(m, dict) else str(m)
                if not model_id:
                    continue
                    
                model_obj = dict(m) if isinstance(m, dict) else {"id": model_id}
                model_obj.setdefault("object", "model")
                model_obj.setdefault("created", 1677610602)
                model_obj.setdefault("owned_by", "llm-cluster")
                
                # Resolve context length from catalog if not reported by backend node
                if "max_model_len" not in model_obj and "context_window" not in model_obj:
                    caps = get_model_capabilities(model_id)
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
                
    # Fallback: if cache is empty, query once in real-time
    if not merged_models:
        logger.info("Models cache empty; performing real-time query on all nodes...")
        tasks = [fetch_node_models(node) for node in CONFIG["nodes"]]
        results = await asyncio.gather(*tasks)
        for models in results:
            for m in models:
                model_id = m.get("id") if isinstance(m, dict) else str(m)
                if not model_id:
                    continue
                model_obj = dict(m) if isinstance(m, dict) else {"id": model_id}
                model_obj.setdefault("object", "model")
                model_obj.setdefault("created", 1677610602)
                model_obj.setdefault("owned_by", "llm-cluster")
                if "max_model_len" not in model_obj and "context_window" not in model_obj:
                    caps = get_model_capabilities(model_id)
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
                    existing_len = merged_models[model_id].get("max_model_len", 0)
                    new_len = model_obj.get("max_model_len", 0)
                    max_len = max(existing_len, new_len)
                    merged_models[model_id]["max_model_len"] = max_len
                    merged_models[model_id]["context_window"] = max_len
                
    return {"object": "list", "data": list(merged_models.values())}


def sanitize_tools(json_data: dict) -> bool:
    """Sanitizes OpenAI tool/function definitions to prevent vLLM/Jinja template type errors.
    Returns True if modifications were made.
    """
    modified = False
    
    # 1. Sanitize "tools" list
    tools = json_data.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if "function" in tool:
                func = tool["function"]
                if not isinstance(func, dict):
                    continue
                
                # Check parameters
                if "parameters" not in func or func["parameters"] is None:
                    func["parameters"] = {
                        "type": "object",
                        "properties": {}
                    }
                    modified = True
                    continue
                    
                parameters = func.get("parameters")
                if isinstance(parameters, str):
                    try:
                        func["parameters"] = json.loads(parameters)
                        parameters = func["parameters"]
                        modified = True
                    except Exception:
                        pass
                
                if not isinstance(parameters, dict):
                    func["parameters"] = {
                        "type": "object",
                        "properties": {}
                    }
                    modified = True
                    continue
                    
                # Ensure properties is a dictionary
                if "properties" not in parameters or not isinstance(parameters.get("properties"), dict) or parameters["properties"] is None:
                    parameters["properties"] = {}
                    modified = True
                    
    # 2. Sanitize legacy "functions" list
    functions = json_data.get("functions")
    if isinstance(functions, list):
        for func in functions:
            if not isinstance(func, dict):
                continue
                
            # Check parameters
            if "parameters" not in func or func["parameters"] is None:
                func["parameters"] = {
                    "type": "object",
                    "properties": {}
                }
                modified = True
                continue
                
            parameters = func.get("parameters")
            if isinstance(parameters, str):
                try:
                    func["parameters"] = json.loads(parameters)
                    parameters = func["parameters"]
                    modified = True
                except Exception:
                    pass
            
            if not isinstance(parameters, dict):
                func["parameters"] = {
                    "type": "object",
                    "properties": {}
                }
                modified = True
                continue
                
            # Ensure properties is a dictionary
            if "properties" not in parameters or not isinstance(parameters.get("properties"), dict) or parameters["properties"] is None:
                parameters["properties"] = {}
                modified = True
                
    return modified


def sanitize_messages(messages: list) -> bool:
    """Sanitizes tool_calls in message history.
    Specifically, converts stringified function arguments in tool_calls to dictionaries
    to prevent Jinja template mapping exceptions.
    Returns True if modifications were made.
    """
    modified = False
    if not isinstance(messages, list):
        return False
        
    for msg in messages:
        if not isinstance(msg, dict):
            continue
            
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function")
                if isinstance(func, dict):
                    arguments = func.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            # Validate it's parseable JSON - vLLM requires arguments to be
                            # a valid JSON string (the OpenAI spec mandates a string, not a dict)
                            json.loads(arguments)
                        except Exception:
                            # arguments is not valid JSON (e.g. empty string, malformed).
                            # Replace with '{}' string so vLLM receives a valid JSON string
                            # instead of crashing with "Expecting value" 400 Bad Request.
                            logger.warning(f"tool_call arguments is not valid JSON (value={repr(arguments)!r}). Replacing with '{{}}' string.")
                            func["arguments"] = "{}"
                            modified = True
                    elif not isinstance(arguments, str):
                        # arguments is already a dict/object - serialize it back to a JSON string
                        try:
                            func["arguments"] = json.dumps(arguments)
                            modified = True
                        except Exception:
                            func["arguments"] = "{}"
                            modified = True
                            
    return modified


def has_image_content(json_data: dict) -> bool:
    """Checks if the request payload contains any multi-modal image content."""
    messages = json_data.get("messages")
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in ("image_url", "image"):
                        return True
                    if "image_url" in item or "image" in item:
                        return True
        elif isinstance(content, dict):
            if content.get("type") in ("image_url", "image") or "image_url" in content or "image" in content:
                return True
    return False


def get_model_capabilities(model_name: str) -> dict:
    """Resolves the capabilities dictionary for a given model from the config catalog.
    If no pattern matches, defaults to all capabilities being True.
    """
    model_name_lower = model_name.lower()
    catalog = CONFIG.get("model_capabilities", [])
    
    for entry in catalog:
        pattern = entry.get("name_pattern", "").lower()
        if pattern and pattern in model_name_lower:
            return {
                "vision": entry.get("vision", True),
                "tool_calling": entry.get("tool_calling", True),
                "structured_output": entry.get("structured_output", True)
            }
            
    # Default assumptions
    return {
        "vision": True,
        "tool_calling": True,
        "structured_output": True
    }


def check_and_reroute_capabilities(json_data: dict) -> bool:
    """Detects if a model was requested with requirements it does not support,
    and rewrites the model parameter to a suitable active fallback model.
    Returns True if the model was rewritten.
    """
    routing_cfg = CONFIG.get("capabilities_routing", {})
    if not routing_cfg.get("enabled", True):
        return False
        
    # 1. Determine request requirements
    needs_vision = has_image_content(json_data)
    needs_tool_calling = "tools" in json_data or "functions" in json_data
    
    # Check structured output requests
    needs_structured = False
    resp_format = json_data.get("response_format")
    if isinstance(resp_format, dict):
        if resp_format.get("type") in ("json_object", "json_schema"):
            needs_structured = True
            
    if not (needs_vision or needs_tool_calling or needs_structured):
        return False
        
    # 2. Resolve requested model capabilities
    requested_model = json_data.get("model", "")
    req_caps = get_model_capabilities(requested_model)
    
    # 3. Detect mismatch
    mismatch = False
    reasons = []
    if needs_vision and not req_caps["vision"]:
        mismatch = True
        reasons.append("vision")
    if needs_tool_calling and not req_caps["tool_calling"]:
        mismatch = True
        reasons.append("tool_calling")
    if needs_structured and not req_caps["structured_output"]:
        mismatch = True
        reasons.append("structured_output")
        
    if not mismatch:
        return False
        
    # 4. Find a suitable active model that satisfies all requirements
    all_active_models = []
    for models in NODE_MODELS_CACHE.values():
        for m in models:
            if m not in all_active_models:
                all_active_models.append(m)
                
    target_model = None
    for active_model in all_active_models:
        caps = get_model_capabilities(active_model)
        
        # Check if active_model satisfies all requirements
        satisfies = True
        if needs_vision and not caps["vision"]:
            satisfies = False
        if needs_tool_calling and not caps["tool_calling"]:
            satisfies = False
        if needs_structured and not caps["structured_output"]:
            satisfies = False
            
        if satisfies:
            target_model = active_model
            break
            
    if target_model and target_model != requested_model:
        logger.info(f"Rerouting request for '{requested_model}' due to missing capabilities ({', '.join(reasons)}). Selected active fallback model: '{target_model}'")
        json_data["model"] = target_model
        return True
        
    return False


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
    
    # Extract client IP and auth info
    client_ip = request.client.host if request.client else "unknown"
    forwarded_for = headers.get("x-forwarded-for")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()
        
    auth_user = parse_auth_user(headers)
    prompt = parse_request_prompt(body)
    trace_id = parse_trace_id(headers)
    
    # Check rate limits (per IP / Auth user)
    client_id = auth_user if auth_user != "anonymous" else client_ip
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded for client '{client_id}'. Max allowed requests per minute reached.")

    # Extract model parameter
    try:
        json_data = await request.json()
        requested_model = json_data.get("model")
        is_stream = json_data.get("stream", False)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Check KV Prefix Cache hash
    prefix_hash = get_prefix_hash(json_data)
    cached_prefix_node = None
    if prefix_hash:
        prefix_cfg = CONFIG.get("prefix_cache_routing", {})
        ttl = prefix_cfg.get("ttl_seconds", 600)
        now = time.time()
        async with CACHE_LOCK:
            if prefix_hash in PREFIX_CACHE:
                c_node, c_time = PREFIX_CACHE[prefix_hash]
                if now - c_time < ttl:
                    cached_prefix_node = c_node
                else:
                    del PREFIX_CACHE[prefix_hash]
        
    if not requested_model:
        raise HTTPException(status_code=400, detail="Missing 'model' parameter in request body")
        
    original_model = requested_model
    body_to_send = body
    
    modified = False
    
    # Sanitize tool/function configurations to protect vLLM templater from Jinja type exceptions
    if "tools" in json_data or "functions" in json_data:
        try:
            if sanitize_tools(json_data):
                modified = True
        except Exception as e:
            logger.warning(f"Error sanitizing tools/functions parameter: {str(e)}")
            
    # Sanitize message history (tool_calls arguments) to prevent Jinja mapping exceptions
    messages = json_data.get("messages")
    if isinstance(messages, list):
        try:
            if sanitize_messages(messages):
                modified = True
        except Exception as e:
            logger.warning(f"Error sanitizing messages history: {str(e)}")
            
    # Check and reroute requests based on model capabilities mismatch fallback rules
    try:
        if check_and_reroute_capabilities(json_data):
            modified = True
            requested_model = json_data["model"]
    except Exception as e:
        logger.warning(f"Error performing capabilities fallback checks: {str(e)}")
            
    if modified:
        body_to_send = json.dumps(json_data).encode("utf-8")
    
    # Find nodes containing this model
    eligible_nodes = []
    async with CACHE_LOCK:
        for node in CONFIG["nodes"]:
            active_models = NODE_MODELS_CACHE.get(node["name"], [])
            active_ids = [m.get("id") if isinstance(m, dict) else str(m) for m in active_models]
            if requested_model in active_ids:
                eligible_nodes.append(node)
                
    if not eligible_nodes:
        # Fallback check: if not in cache, query in real time in case model was just loaded
        logger.info(f"Model '{requested_model}' not found in cache. Querying nodes in real-time...")
        for node in CONFIG["nodes"]:
            models = await fetch_node_models(node)
            active_ids = [m.get("id") if isinstance(m, dict) else str(m) for m in models]
            if requested_model in active_ids:
                eligible_nodes.append(node)
                async with CACHE_LOCK:
                    NODE_MODELS_CACHE[node["name"]] = models
                    
    # Check if we should auto-rewrite the model parameter to an active fallback model
    if not eligible_nodes:
        all_active_models = []
        async with CACHE_LOCK:
            for models in NODE_MODELS_CACHE.values():
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
                
            # Re-find nodes containing this fallback model
            async with CACHE_LOCK:
                for node in CONFIG["nodes"]:
                    active_models = NODE_MODELS_CACHE.get(node["name"], [])
                    active_ids = [m.get("id") if isinstance(m, dict) else str(m) for m in active_models]
                    if requested_model in active_ids:
                        eligible_nodes.append(node)
                        
    if not eligible_nodes:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{original_model}' is not currently loaded on any online cluster nodes, and no active fallback models are currently served by the cluster."
        )
        
    # Load balance based on configured mode
    routing_config = CONFIG.get("routing", {})
    routing_mode = routing_config.get("mode", "smart").lower()
    sticky_header = routing_config.get("sticky_header", "x-session-id").lower()

    # Prefix Cache Routing check: if this prefix hash was previously served by an active eligible node, prefer that node!
    prefix_matched_node = None
    if cached_prefix_node:
        for n in eligible_nodes:
            if n["name"] == cached_prefix_node:
                prefix_matched_node = n
                break

    if prefix_matched_node:
        selected_node = prefix_matched_node
        logger.info(f"Prefix KV-cache hit for hash '{prefix_hash}'. Routing to warm cache node '{selected_node['name']}'")
    elif routing_mode == "smart":
        # Check if a sticky key is present. If yes, route stickily. If not, route to least loaded.
        sticky_key = headers.get(sticky_header)
        if not sticky_key:
            for h_name, h_val in headers.items():
                if h_name.lower() == sticky_header:
                    sticky_key = h_val
                    break

        if sticky_key:
            sorted_nodes = sorted(eligible_nodes, key=lambda n: n["name"])
            hash_val = zlib.crc32(sticky_key.encode("utf-8"))
            selected_node = sorted_nodes[hash_val % len(sorted_nodes)]
            logger.info(f"Smart routing selected sticky node {selected_node['name']} for key '{sticky_key}'")
        else:


            # Context Window & Thermal-aware routing
            # Estimate requested token length (approx 3.2 chars per token for safe estimation + max_tokens requested)
            est_prompt_tokens = int(len(prompt) / 3.2) if prompt else 0
            max_gen_tokens = json_data.get("max_tokens", 2048) if isinstance(json_data, dict) else 2048
            est_total_request_len = est_prompt_tokens + max_gen_tokens

            candidate_nodes = list(eligible_nodes)
            
            # Filter nodes by context window capacity if max_model_len reported
            context_capable_nodes = []
            for n in candidate_nodes:
                node_models = NODE_MODELS_CACHE.get(n["name"], [])
                node_ctx = 0
                for m in node_models:
                    m_id = m.get("id") if isinstance(m, dict) else str(m)
                    if m_id == requested_model:
                        node_ctx = m.get("max_model_len", m.get("context_window", 0)) if isinstance(m, dict) else 0
                        break
                # If node reports max_model_len and it's smaller than estimated request, skip node
                if node_ctx > 0 and est_total_request_len > node_ctx:
                    logger.info(f"Skipping node '{n['name']}' for '{requested_model}': est request len ({est_total_request_len}) exceeds node context limit ({node_ctx})")
                    continue
                context_capable_nodes.append(n)
                
            if context_capable_nodes:
                candidate_nodes = context_capable_nodes
            else:
                logger.warning(f"No active node for '{requested_model}' supports est context length ({est_total_request_len}). Routing to all eligible nodes.")

            thermal_cfg = CONFIG.get("thermal_routing", {})
            thermal_enabled = thermal_cfg.get("enabled", True)
            max_temp = thermal_cfg.get("max_temp_celsius", 82.0)

            if thermal_enabled:
                cool_nodes = [n for n in candidate_nodes if NODE_TEMP_CACHE.get(n["name"], 0) < max_temp]
                if cool_nodes:
                    candidate_nodes = cool_nodes
                else:
                    logger.warning(f"All candidate nodes for '{requested_model}' exceed thermal limit ({max_temp}°C). Routing to coolest available.")

            # Sort by active requests, then node priority (lower = higher priority), then lower temperature
            selected_node = min(
                candidate_nodes,
                key=lambda n: (
                    ACTIVE_REQUESTS.get(n["name"], 0),
                    n.get("priority", 1),
                    NODE_TEMP_CACHE.get(n["name"], 0)
                )
            )
            active_count = ACTIVE_REQUESTS.get(selected_node["name"], 0)
            node_temp = NODE_TEMP_CACHE.get(selected_node["name"])
            temp_info = f", temp: {node_temp:.1f}°C" if node_temp is not None else ""
            logger.info(f"Smart routing selected node {selected_node['name']} (active: {active_count}{temp_info})")

    # Record prefix cache location for future requests
    if prefix_hash and selected_node:
        async with CACHE_LOCK:
            PREFIX_CACHE[prefix_hash] = (selected_node["name"], time.time())


    elif routing_mode == "sticky":
        # Find sticky key: session-id header -> auth-user -> client-ip
        sticky_key = headers.get(sticky_header)
        if not sticky_key:
            for h_name, h_val in headers.items():
                if h_name.lower() == sticky_header:
                    sticky_key = h_val
                    break
        if not sticky_key:
            sticky_key = auth_user if auth_user != "anonymous" else client_ip

        # Sort eligible nodes to ensure consistent order
        sorted_nodes = sorted(eligible_nodes, key=lambda n: n["name"])
        hash_val = zlib.crc32(sticky_key.encode("utf-8"))
        selected_node = sorted_nodes[hash_val % len(sorted_nodes)]
        logger.info(f"Sticky routing selected node {selected_node['name']} for key '{sticky_key}'")
    else:
        selected_node = random.choice(eligible_nodes)
        logger.info(f"Randomly selected node {selected_node['name']} for model '{requested_model}'")
    
    return await forward_request(
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


@app.get("/health")
async def health():
    """Liveness check for the router proxy itself."""
    return {"status": "healthy"}
