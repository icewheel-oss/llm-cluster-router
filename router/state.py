# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Shared mutable state + config hot-reload. Every other module reads
CONFIG/NODE_MODELS_CACHE/NODE_TEMP_CACHE/ACTIVE_REQUESTS via `from router
import state` and accesses `state.CONFIG` etc, rather than importing the
names directly -- that's what lets `reload_config_if_changed()` swap in a
whole new CONFIG dict and have every module see it immediately, without
each of them needing its own reload logic.
"""
import asyncio
import os
from typing import Any, Dict, List

import httpx
import yaml

from router.logging_setup import logger

CONFIG: Dict[str, Any] = {}
CLIENT: httpx.AsyncClient = None
LAST_CONFIG_MTIME: float = 0.0

# Cache of active models per node: {node_name: [model_obj_1, model_obj_2, ...]}
NODE_MODELS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
NODE_TEMP_CACHE: Dict[str, float] = {}
ACTIVE_REQUESTS: Dict[str, int] = {}

CACHE_LOCK = asyncio.Lock()


def increment_active_requests(node_name: str) -> None:
    """Called by proxy.forward_request when a request starts -- the
    "smart" strategy reads ACTIVE_REQUESTS to prefer less-loaded nodes."""
    ACTIVE_REQUESTS[node_name] = ACTIVE_REQUESTS.get(node_name, 0) + 1


def decrement_active_requests(node_name: str) -> None:
    """Called when a request finishes (audit_log.stream_and_log) or fails
    to even start (proxy.forward_request's error path)."""
    ACTIVE_REQUESTS[node_name] = max(0, ACTIVE_REQUESTS.get(node_name, 0) - 1)


def load_config() -> Dict[str, Any]:
    """Reads config.yaml (path from CONFIG_PATH env var, default
    "config.yaml") fresh from disk. Called once at startup (router/app.py's
    lifespan) and again by reload_config_if_changed() -- this function
    itself does no comparison/caching, it always re-reads."""
    global LAST_CONFIG_MTIME
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    LAST_CONFIG_MTIME = os.path.getmtime(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def reload_config_if_changed() -> bool:
    """Reloads CONFIG dynamically if config.yaml's modification time changed.
    Called from the background polling loop AND from the manual
    POST /_router/reload endpoint -- this is the single mechanism behind
    both, so hot-reload behavior (and its normalization defaults) stays
    identical regardless of which one triggered it.
    """
    global CONFIG, LAST_CONFIG_MTIME, ACTIVE_REQUESTS
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    if not os.path.exists(config_path):
        return False
    current_mtime = os.path.getmtime(config_path)
    if current_mtime != LAST_CONFIG_MTIME:
        try:
            new_config = load_config()
            new_config.setdefault("timeouts", {})
            new_config["timeouts"].setdefault("primary", 1.0)
            new_config["timeouts"].setdefault("fallback", 3.0)
            new_config["timeouts"].setdefault("request", 120.0)
            new_config.setdefault("general_settings", {})
            new_config.setdefault("thermal_routing", {})
            new_config["thermal_routing"].setdefault("enabled", True)
            new_config["thermal_routing"].setdefault("warning_temp_celsius", 70.0)
            new_config["thermal_routing"].setdefault("critical_temp_celsius", 80.0)
            new_config["thermal_routing"].setdefault("thermal_priority_over_kv_cache", True)

            CONFIG = new_config
            for node in CONFIG.get("nodes", []):
                if node["name"] not in ACTIVE_REQUESTS:
                    ACTIVE_REQUESTS[node["name"]] = 0
            logger.info("⚡ Configuration hot-reloaded dynamically from config.yaml!")
            return True
        except Exception as e:
            logger.error(f"Failed to hot-reload configuration: {e}")
    return False
