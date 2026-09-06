# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import asyncio
import base64
import json
import os
import time
from datetime import datetime
from typing import Dict, Optional

import httpx

from router import state
from router.logging_setup import audit_logger, logger

LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", "5044"))


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


def parse_trace_id(headers: Dict[str, str]) -> Optional[str]:
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


async def send_log_to_logstash(log_entry: dict) -> None:
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

        state.decrement_active_requests(node_name)
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
        audit_logger.info(f"AUDIT_LOG: {json.dumps(log_entry)}")
        asyncio.create_task(send_log_to_logstash(log_entry))
