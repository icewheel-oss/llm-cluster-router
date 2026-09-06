# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""The actual reverse-proxy hop: once router/app.py has picked a node
(via prefix affinity or a routing strategy), this is what sends the
request there, fails over from the node's primary interface to its
backup if configured, and wraps the response in the streaming
audit-logging wrapper (router/audit_log.stream_and_log)."""
import time
from typing import Dict

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from router import audit_log, state
from router.logging_setup import logger


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

    state.increment_active_requests(node["name"])
    success = False
    try:
        # Try Primary
        try:
            url = f"{primary_base}/{path}"
            logger.info(f"Forwarding {method} request to primary node {node['name']} at {url}")
            req = state.CLIENT.build_request(method, url, headers=headers_to_send, content=content, timeout=None)
            resp = await state.CLIENT.send(req, stream=True)
            success = True
            return StreamingResponse(
                audit_log.stream_and_log(
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
                req = state.CLIENT.build_request(method, url, headers=headers_to_send, content=content, timeout=None)
                resp = await state.CLIENT.send(req, stream=True)
                success = True
                return StreamingResponse(
                    audit_log.stream_and_log(
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
            state.decrement_active_requests(node["name"])
