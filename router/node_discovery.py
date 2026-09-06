# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

from typing import Any, Dict, List

import httpx

from router import state
from router.logging_setup import logger


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
        resp = await state.CLIENT.get(url, headers=req_headers, timeout=state.CONFIG["timeouts"]["primary"])
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
        logger.warning(f"Primary endpoint failed for {node['name']} ({str(e)}). Trying backup...")

    # Try Backup
    if backup:
        try:
            url = f"{backup.rstrip('/')}/models"
            logger.debug(f"Querying models from backup node {node['name']} at {url}")
            resp = await state.CLIENT.get(url, headers=req_headers, timeout=state.CONFIG["timeouts"]["fallback"])
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
