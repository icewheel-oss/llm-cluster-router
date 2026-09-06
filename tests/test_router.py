# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import os
import time
import yaml
import asyncio
import pytest
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from unittest.mock import AsyncMock, patch, MagicMock

from router import app as app_module
from router import audit_log, model_matching, node_discovery, prefix_cache, proxy, rate_limit, sanitize, state, strategies, thermal
from router.app import app

# Apply asyncio marker to all tests in this file
pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def setup_config():
    """Initializes a mock configuration before each test."""
    state.CONFIG = {
        "timeouts": {
            "primary": 0.1,
            "fallback": 0.2,
            "request": 1.0
        },
        "general_settings": {
            "health_check_interval": 10
        },
        "nodes": [
            {
                "name": "node-1",
                "primary": "http://192.168.86.221:8000/v1",
                "backup": "http://192.168.86.211:8000/v1"
            },
            {
                "name": "node-4",
                "primary": "http://192.168.86.224:8000/v1",
                "backup": "http://192.168.86.214:8000/v1"
            }
        ]
    }
    state.NODE_MODELS_CACHE = {}
    prefix_cache.PREFIX_CACHE.clear()


async def test_health():
    """Test health check endpoint."""
    resp = await app_module.health()
    assert resp == {"status": "healthy"}


async def test_fetch_node_models_primary_success(mocker):
    """Test fetch_node_models succeeds on primary Ethernet path."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"id": "model-a"}]
    }
    mock_client.get.return_value = mock_response
    state.CLIENT = mock_client

    node = state.CONFIG["nodes"][0]
    models = await node_discovery.fetch_node_models(node)

    assert models == [{"id": "model-a"}]
    mock_client.get.assert_called_once_with(
        "http://192.168.86.221:8000/v1/models",
        headers={},
        timeout=0.1
    )


async def test_fetch_node_models_failover_to_backup(mocker):
    """Test fetch_node_models fails on primary and successfully fails over to backup WiFi path."""
    mock_client = AsyncMock()

    mock_client.get.side_effect = [
        httpx.TimeoutException("Primary timed out"),
        MagicMock(status_code=200, json=lambda: {"data": [{"id": "model-b"}]})
    ]
    state.CLIENT = mock_client

    node = state.CONFIG["nodes"][0]
    models = await node_discovery.fetch_node_models(node)

    assert models == [{"id": "model-b"}]
    assert mock_client.get.call_count == 2

    mock_client.get.assert_any_call("http://192.168.86.221:8000/v1/models", headers={}, timeout=0.1)
    mock_client.get.assert_any_call("http://192.168.86.211:8000/v1/models", headers={}, timeout=0.2)


async def test_fetch_node_models_all_fail(mocker):
    """Test fetch_node_models returns empty list if both primary and backup paths fail."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.RequestError("Network error")
    state.CLIENT = mock_client

    node = state.CONFIG["nodes"][0]
    models = await node_discovery.fetch_node_models(node)

    assert models == []


async def test_get_models_from_cache():
    """Test get_models returns deduplicated models lists from internal cache."""
    state.NODE_MODELS_CACHE = {
        "node-1": ["model-a", "model-b"],
        "node-4": ["model-b", "model-c"]
    }

    resp = await app_module.get_models()
    assert resp["object"] == "list"

    model_ids = {m["id"] for m in resp["data"]}
    assert model_ids == {"model-a", "model-b", "model-c"}


async def test_forward_request_primary_success(mocker):
    """Test forwarding requests successfully routes to primary node."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aiter_bytes.return_value = AsyncIterator([b"response-chunk"])

    mock_client.build_request.return_value = "mocked-request"
    mock_client.send.return_value = mock_response
    state.CLIENT = mock_client

    node = state.CONFIG["nodes"][0]
    resp = await proxy.forward_request(
        node=node,
        path="chat/completions",
        method="POST",
        headers={"content-type": "application/json", "Authorization": "Bearer key"},
        content=b'{"test": 1}',
        client_ip="127.0.0.1",
        auth_user="demouser",
        requested_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        prompt="hi",
        is_stream=False
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"

    mock_client.build_request.assert_called_once_with(
        "POST",
        "http://192.168.86.221:8000/v1/chat/completions",
        headers={"content-type": "application/json", "Authorization": "Bearer key"},
        content=b'{"test": 1}',
        timeout=None
    )


async def test_forward_request_failover_to_backup(mocker):
    """Test forwarding requests falls back to backup WiFi if primary Ethernet times out."""
    mock_client = AsyncMock()

    mock_response = MagicMock(status_code=200, headers={}, aiter_bytes=lambda: AsyncIterator([b"ok"]))
    mock_client.send.side_effect = [
        httpx.TimeoutException("Timeout"),
        mock_response
    ]

    mock_client.build_request.side_effect = ["req-1", "req-2"]
    state.CLIENT = mock_client

    node = state.CONFIG["nodes"][0]
    resp = await proxy.forward_request(
        node=node,
        path="chat/completions",
        method="POST",
        headers={},
        content=b"",
        client_ip="127.0.0.1",
        auth_user="anonymous",
        requested_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        prompt="hi",
        is_stream=False
    )

    assert resp.status_code == 200
    assert mock_client.send.call_count == 2

    mock_client.build_request.assert_any_call("POST", "http://192.168.86.221:8000/v1/chat/completions", headers={}, content=b"", timeout=None)
    mock_client.build_request.assert_any_call("POST", "http://192.168.86.211:8000/v1/chat/completions", headers={}, content=b"", timeout=None)


# Helper class to mock asynchronous chunk iteration in StreamingResponse
class AsyncIterator:
    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration


async def test_handle_llm_request_non_streaming(mocker):
    """Test handle_llm_request correctly routes and returns non-streaming completions response."""
    state.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "node-4": []
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{"id":"chatcmpl-123","object":"chat.completion","choices":[{"message":{"content":"Hello!"}}]}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json", "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    resp = await app_module.handle_llm_request(mock_request)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"

    chunks = [chunk async for chunk in resp.body_iterator]
    assert b"chatcmpl-123" in chunks[0]

    proxy.forward_request.assert_called_once()
    args, kwargs = proxy.forward_request.call_args
    assert kwargs["node"]["name"] == "node-1"
    assert kwargs["path"] == "chat/completions"
    assert kwargs["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


async def test_handle_llm_request_streaming(mocker):
    """Test handle_llm_request correctly forwards streaming requests and chunks event stream."""
    state.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }

    sse_data = [
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
        b'data: [DONE]\n\n'
    ]
    mock_resp = StreamingResponse(
        AsyncIterator(sse_data),
        status_code=200,
        headers={"content-type": "text/event-stream"}
    )
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "stream": true}')
    mock_request.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "stream": True})

    resp = await app_module.handle_llm_request(mock_request)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream"

    chunks = [chunk async for chunk in resp.body_iterator]
    assert chunks == sse_data


async def test_handle_llm_request_model_rewriting(mocker):
    """Test that when a client requests a model that is not loaded, the gateway rewrites it to an active model fallback."""
    import json
    state.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{"id":"chatcmpl-123","object":"chat.completion","choices":[{"message":{"content":"Hello!"}}]}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.body = AsyncMock(return_value=b'{"model": "model-not-found"}')
    mock_request.json = AsyncMock(return_value={"model": "model-not-found"})

    resp = await app_module.handle_llm_request(mock_request)
    assert resp.status_code == 200

    proxy.forward_request.assert_called_once()
    args, kwargs = proxy.forward_request.call_args
    assert kwargs["node"]["name"] == "node-1"
    assert kwargs["requested_model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert kwargs["original_model"] == "model-not-found"

    rewritten_body = json.loads(kwargs["content"].decode("utf-8"))
    assert rewritten_body["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"


async def test_handle_llm_request_sticky_routing(mocker):
    """Test that requests with the same sticky key are routed consistently to the same node in sticky mode."""
    state.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "node-2": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }
    state.CONFIG["routing"] = {
        "mode": "sticky",
        "sticky_header": "x-session-id"
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    mock_request_1 = MagicMock(spec=Request)
    mock_request_1.url = MagicMock()
    mock_request_1.url.path = "/v1/chat/completions"
    mock_request_1.method = "POST"
    mock_request_1.headers = {"content-type": "application/json", "x-session-id": "session-1"}
    mock_request_1.client = MagicMock()
    mock_request_1.client.host = "127.0.0.1"
    mock_request_1.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request_1.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    mock_request_2 = MagicMock(spec=Request)
    mock_request_2.url = MagicMock()
    mock_request_2.url.path = "/v1/chat/completions"
    mock_request_2.method = "POST"
    mock_request_2.headers = {"content-type": "application/json", "x-session-id": "session-1"}
    mock_request_2.client = MagicMock()
    mock_request_2.client.host = "127.0.0.1"
    mock_request_2.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request_2.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    resp1 = await app_module.handle_llm_request(mock_request_1)
    assert resp1.status_code == 200
    args1, kwargs1 = proxy.forward_request.call_args
    node_selected_1 = kwargs1["node"]["name"]

    proxy.forward_request.reset_mock()
    resp2 = await app_module.handle_llm_request(mock_request_2)
    assert resp2.status_code == 200
    args2, kwargs2 = proxy.forward_request.call_args
    node_selected_2 = kwargs2["node"]["name"]

    assert node_selected_1 == node_selected_2


async def test_handle_llm_request_smart_routing(mocker):
    """Test that smart routing routes stickily when session ID is present, and to the least loaded when absent."""
    state.NODE_MODELS_CACHE = {
        "node-1": [{"id": "deepseek-ai/DeepSeek-V4-Flash-0731", "max_model_len": 131072}],
        "node-2": [{"id": "deepseek-ai/DeepSeek-V4-Flash-0731", "max_model_len": 131072}]
    }
    state.CONFIG["nodes"] = [
        {"name": "node-1", "primary": "http://192.168.86.221:8000/v1"},
        {"name": "node-2", "primary": "http://192.168.86.222:8000/v1"}
    ]
    state.CONFIG["routing"] = {
        "mode": "smart",
        "sticky_header": "x-session-id"
    }
    state.ACTIVE_REQUESTS = {
        "node-1": 3,
        "node-2": 0
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    mock_request_no_session = MagicMock(spec=Request)
    mock_request_no_session.url = MagicMock()
    mock_request_no_session.url.path = "/v1/chat/completions"
    mock_request_no_session.method = "POST"
    mock_request_no_session.headers = {"content-type": "application/json"}
    mock_request_no_session.client = MagicMock()
    mock_request_no_session.client.host = "127.0.0.1"
    mock_request_no_session.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request_no_session.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    resp1 = await app_module.handle_llm_request(mock_request_no_session)
    assert resp1.status_code == 200
    args1, kwargs1 = proxy.forward_request.call_args
    assert kwargs1["node"]["name"] == "node-2"


@pytest.mark.parametrize("mode", ["smart", "sticky", "random"])
async def test_prefix_cache_recorded_regardless_of_strategy(mocker, mode):
    """Regression test for the bug this refactor fixes: previously, only
    "smart" mode ever recorded a PREFIX_CACHE entry (a misplaced elif in
    the old inline handle_llm_request meant "sticky" and the random
    fallback never seeded the affinity map at all). Every strategy must
    record an entry so a LATER request with the same prefix hash can
    actually find and use it."""
    state.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "node-4": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }
    state.CONFIG["routing"] = {"mode": mode, "sticky_header": "x-session-id"}
    state.ACTIVE_REQUESTS = {"node-1": 0, "node-4": 0}

    mock_resp = StreamingResponse(AsyncIterator([b'{}']), status_code=200, headers={})
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    long_system_prompt = "You are a helpful assistant. " * 5  # comfortably over min_prefix_length
    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    payload = {
        "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
        "messages": [
            {"role": "system", "content": long_system_prompt},
            {"role": "user", "content": "hi"}
        ]
    }
    import json as _json
    mock_request.body = AsyncMock(return_value=_json.dumps(payload).encode())
    mock_request.json = AsyncMock(return_value=payload)

    expected_hash = prefix_cache.get_prefix_hash(payload)
    assert expected_hash, "test setup: prompt should be long enough to produce a real hash"
    assert expected_hash not in prefix_cache.PREFIX_CACHE

    resp = await app_module.handle_llm_request(mock_request)
    assert resp.status_code == 200

    assert expected_hash in prefix_cache.PREFIX_CACHE, (
        f"routing.mode={mode!r} did not record a PREFIX_CACHE entry"
    )
    args, kwargs = proxy.forward_request.call_args
    recorded_node, _ = prefix_cache.PREFIX_CACHE[expected_hash]
    assert recorded_node == kwargs["node"]["name"]


def test_stage2_affinity_bypasses_when_context_too_small():
    """A warm node whose context window is smaller than the estimated
    request length must be bypassed, even if it's thermally fine --
    closes the gap where Stage 2 used to skip the check the "smart"
    strategy's own fallback path already did."""
    state.NODE_MODELS_CACHE = {
        "small-node": [{"id": "test-model", "max_model_len": 4096}],
        "big-node": [{"id": "test-model", "max_model_len": 262144}],
    }
    state.NODE_TEMP_CACHE = {"small-node": 40.0}
    state.ACTIVE_REQUESTS = {"small-node": 0, "big-node": 0}
    eligible_nodes = [{"name": "small-node"}, {"name": "big-node"}]

    result = thermal.apply_stage2_prefix_affinity(
        "small-node", eligible_nodes, "test-model", est_request_len=8000
    )
    assert result is None


def test_stage2_affinity_returns_node_when_context_fits():
    """Regression check: a warm node that DOES have enough context, and
    isn't hot or overloaded, must still be returned directly (the whole
    point of prefix affinity)."""
    state.NODE_MODELS_CACHE = {
        "small-node": [{"id": "test-model", "max_model_len": 4096}],
    }
    state.NODE_TEMP_CACHE = {"small-node": 40.0}
    state.ACTIVE_REQUESTS = {"small-node": 0}
    eligible_nodes = [{"name": "small-node"}]

    result = thermal.apply_stage2_prefix_affinity(
        "small-node", eligible_nodes, "test-model", est_request_len=2000
    )
    assert result == {"name": "small-node"}


def test_stage2_affinity_bypasses_on_load_imbalance():
    """A warm node running significantly more active requests than
    another eligible node must be bypassed -- this is the actual fix for
    the real imbalance found live (one consistently-cool node absorbing
    a hugely disproportionate share of traffic because it never tripped
    the thermal check)."""
    state.CONFIG.setdefault("thermal_routing", {})
    state.CONFIG["thermal_routing"]["max_affinity_load_imbalance"] = 3
    state.NODE_MODELS_CACHE = {
        "warm-node": [{"id": "test-model", "max_model_len": 262144}],
        "idle-node": [{"id": "test-model", "max_model_len": 262144}],
    }
    state.NODE_TEMP_CACHE = {"warm-node": 40.0}
    state.ACTIVE_REQUESTS = {"warm-node": 8, "idle-node": 1}
    eligible_nodes = [{"name": "warm-node"}, {"name": "idle-node"}]

    result = thermal.apply_stage2_prefix_affinity(
        "warm-node", eligible_nodes, "test-model", est_request_len=2000
    )
    assert result is None


def test_stage2_affinity_load_imbalance_disabled_by_zero():
    """max_affinity_load_imbalance: 0 must disable the check entirely --
    the warm node is returned even with a large active-request gap."""
    state.CONFIG.setdefault("thermal_routing", {})
    state.CONFIG["thermal_routing"]["max_affinity_load_imbalance"] = 0
    state.NODE_MODELS_CACHE = {
        "warm-node": [{"id": "test-model", "max_model_len": 262144}],
        "idle-node": [{"id": "test-model", "max_model_len": 262144}],
    }
    state.NODE_TEMP_CACHE = {"warm-node": 40.0}
    state.ACTIVE_REQUESTS = {"warm-node": 20, "idle-node": 0}
    eligible_nodes = [{"name": "warm-node"}, {"name": "idle-node"}]

    result = thermal.apply_stage2_prefix_affinity(
        "warm-node", eligible_nodes, "test-model", est_request_len=2000
    )
    assert result == {"name": "warm-node"}


def test_strategy_registry_has_builtins():
    """The three built-in strategies self-register just by importing the
    router.strategies package (see its __init__.py for how)."""
    names = strategies.registered_strategy_names()
    assert {"random", "sticky", "smart"}.issubset(set(names))


def test_strategy_registry_unknown_name_raises():
    with pytest.raises(ValueError):
        strategies.get_strategy("not-a-real-strategy")


async def test_openai_compatible_errors():
    """Test that standard HTTPExceptions are formatted as standard OpenAI error payloads."""
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/v1/non-existent-route")
    assert resp.status_code == 404
    error_json = resp.json()
    assert "error" in error_json
    assert "message" in error_json["error"]
    assert error_json["error"]["type"] == "invalid_request_error"
    assert error_json["error"]["code"] == "404"


def test_sanitize_tools():
    """Test tool calling schemas sanitization rules."""
    data_missing_params = {
        "model": "test-model",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather"
                }
            }
        ]
    }
    modified1 = sanitize.sanitize_tools(data_missing_params)
    assert modified1 is True
    func1 = data_missing_params["tools"][0]["function"]
    assert "parameters" in func1
    assert func1["parameters"]["properties"] == {}

    data_missing_props = {
        "model": "test-model",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object"
                    }
                }
            }
        ]
    }
    modified2 = sanitize.sanitize_tools(data_missing_props)
    assert modified2 is True
    func2 = data_missing_props["tools"][0]["function"]
    assert func2["parameters"]["properties"] == {}

    data_string_params = {
        "model": "test-model",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": '{"type": "object", "properties": {"city": {"type": "string"}}}'
                }
            }
        ]
    }
    modified3 = sanitize.sanitize_tools(data_string_params)
    assert modified3 is True
    func3 = data_string_params["tools"][0]["function"]
    assert isinstance(func3["parameters"], dict)
    assert func3["parameters"]["properties"]["city"]["type"] == "string"

    data_legacy_functions = {
        "model": "test-model",
        "functions": [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object"
                }
            }
        ]
    }
    modified4 = sanitize.sanitize_tools(data_legacy_functions)
    assert modified4 is True
    func4 = data_legacy_functions["functions"][0]
    assert func4["parameters"]["properties"] == {}


def test_sanitize_messages():
    """Test that tool call arguments in messages are sanitized as valid JSON strings."""
    messages = [
        {
            "role": "user",
            "content": "What is the weather?"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": ""
                    }
                }
            ]
        }
    ]
    modified = sanitize.sanitize_messages(messages)
    assert modified is True
    args = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert args == "{}"


def test_has_image_content():
    """Test multimodal image detection in payloads."""
    data_text = {
        "messages": [{"role": "user", "content": "hello world"}]
    }
    assert model_matching.has_image_content(data_text) is False

    data_image = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
                ]
            }
        ]
    }
    assert model_matching.has_image_content(data_image) is True


def test_check_and_reroute_capabilities():
    """Test that requests matching capability mismatch fallback rules are rerouted correctly."""
    state.CONFIG["capabilities_routing"] = {"enabled": True}
    state.CONFIG["model_capabilities"] = [
        {"name_pattern": "r1", "vision": False, "tool_calling": False, "structured_output": False},
        {"name_pattern": "qwen", "vision": True, "tool_calling": True, "structured_output": True},
        {"name_pattern": "deepseek", "vision": False, "tool_calling": True, "structured_output": True}
    ]
    state.NODE_MODELS_CACHE = {
        "node-1": ["DeepSeek-V4-Flash-0731"],
        "node-3": ["Qwen/Qwen3.8-27B-FP8"],
        "node-4": ["DeepSeek-R1-Distill-Q8"]
    }

    payload1 = {
        "model": "DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": "hi"}]
    }
    assert model_matching.check_and_reroute_capabilities(payload1) is False
    assert payload1["model"] == "DeepSeek-V4-Flash-0731"

    payload2 = {
        "model": "DeepSeek-V4-Flash-0731",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]
            }
        ]
    }
    assert model_matching.check_and_reroute_capabilities(payload2) is True
    assert payload2["model"] == "Qwen/Qwen3.8-27B-FP8"

    payload3 = {
        "model": "DeepSeek-R1-Distill-Q8",
        "messages": [{"role": "user", "content": "Get weather"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}
            }
        ]
    }
    assert model_matching.check_and_reroute_capabilities(payload3) is True
    assert payload3["model"] == "DeepSeek-V4-Flash-0731"


async def test_stream_and_log_cancelled(mocker):
    """Test that client cancellation terminates the backend streamed response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}

    async def mock_aiter_bytes():
        raise asyncio.CancelledError()
        yield b""  # make it a generator

    mock_response.aiter_bytes = mock_aiter_bytes

    aclose_called = False

    async def mock_aclose():
        nonlocal aclose_called
        aclose_called = True

    mock_response.aclose = mock_aclose

    gen = audit_log.stream_and_log(
        resp=mock_response,
        node_name="node-1",
        method="POST",
        path="v1/chat/completions",
        client_ip="127.0.0.1",
        auth_user="demouser",
        requested_model="model-a",
        prompt="hello",
        start_time=123.45,
        is_stream=True
    )

    with pytest.raises(asyncio.CancelledError):
        async for chunk in gen:
            pass

    assert aclose_called is True


def test_prefix_cache_routing():
    """Test system prompt prefix hashing and cache stickiness."""
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in stock news analysis and portfolio risk assessment."},
            {"role": "user", "content": "Analyze AAPL"}
        ],
        "model": "deepseek-v4"
    }
    hash_val = prefix_cache.get_prefix_hash(payload)
    assert len(hash_val) > 0

    prefix_cache.PREFIX_CACHE[hash_val] = ("node-3", time.time())
    c_node, c_time = prefix_cache.PREFIX_CACHE.get(hash_val, (None, 0))
    assert c_node == "node-3"


def test_rate_limiting():
    """Test client rate limiting quota enforcement."""
    state.CONFIG["rate_limiting"] = {"enabled": True, "requests_per_minute": 2}
    rate_limit.RATE_LIMIT_CACHE.clear()

    client = "user-test"
    assert rate_limit.check_rate_limit(client) is True
    assert rate_limit.check_rate_limit(client) is True
    assert rate_limit.check_rate_limit(client) is False


async def test_fetch_node_models_custom_headers(mocker):
    """Test fetch_node_models sends custom node headers if specified in config."""
    mock_client = AsyncMock()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"data": [{"id": "peer-model", "max_model_len": 262144}]}
    mock_client.get.return_value = mock_response
    state.CLIENT = mock_client

    node = {
        "name": "peer-node",
        "primary": "https://example.invalid/v1",
        "headers": {"Authorization": "Bearer peer-token-secret"}
    }
    models = await node_discovery.fetch_node_models(node)

    assert models == [{"id": "peer-model", "max_model_len": 262144}]
    mock_client.get.assert_called_once_with(
        "https://example.invalid/v1/models",
        headers={"Authorization": "Bearer peer-token-secret"},
        timeout=0.1
    )


async def test_get_models_max_context_aggregation():
    """Test /v1/models aggregates maximum context_window across multiple nodes."""
    state.NODE_MODELS_CACHE = {
        "node-local": [{"id": "Qwen/Qwen3.8-27B-FP8", "max_model_len": 131072}],
        "node-peer": [{"id": "Qwen/Qwen3.8-27B-FP8", "max_model_len": 262144}]
    }

    res = await app_module.get_models()
    assert "data" in res
    assert len(res["data"]) == 1
    model_info = res["data"][0]
    assert model_info["id"] == "Qwen/Qwen3.8-27B-FP8"
    assert model_info["max_model_len"] == 262144
    assert model_info["context_window"] == 262144


async def test_context_window_routing_filter(mocker):
    """Test smart routing excludes small-context nodes when estimated request token length exceeds node limit."""
    state.NODE_MODELS_CACHE = {
        "node-1": [{"id": "Qwen/Qwen3.8-27B-FP8", "max_model_len": 131072}],
        "node-2": [{"id": "Qwen/Qwen3.8-27B-FP8", "max_model_len": 262144}]
    }
    state.CONFIG["nodes"] = [
        {"name": "node-1", "primary": "http://192.168.86.221:8000/v1"},
        {"name": "node-2", "primary": "http://192.168.86.222:8000/v1"}
    ]
    state.CONFIG["routing"] = {"mode": "smart"}
    state.ACTIVE_REQUESTS = {"node-1": 0, "node-2": 0}

    mock_resp = StreamingResponse(
        AsyncIterator([b'{}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("router.proxy.forward_request", return_value=mock_resp)

    large_prompt = "x" * 600000
    mock_req = MagicMock(spec=Request)
    mock_req.url = MagicMock()
    mock_req.url.path = "/v1/chat/completions"
    mock_req.method = "POST"
    mock_req.headers = {"content-type": "application/json"}
    mock_req.client = MagicMock()
    mock_req.client.host = "127.0.0.1"
    mock_req.body = AsyncMock(return_value=f'{{"model": "Qwen/Qwen3.8-27B-FP8", "messages": [{{"role": "user", "content": "{large_prompt}"}}]}}'.encode())
    mock_req.json = AsyncMock(return_value={"model": "Qwen/Qwen3.8-27B-FP8", "messages": [{"role": "user", "content": large_prompt}]})

    resp = await app_module.handle_llm_request(mock_req)
    assert resp.status_code == 200
    args, kwargs = proxy.forward_request.call_args
    assert kwargs["node"]["name"] == "node-2"


async def test_reload_config_if_changed(tmp_path, monkeypatch):
    """Test reload_config_if_changed reloads configuration when config file modification time updates."""
    config_file = tmp_path / "config.yaml"
    initial_content = {
        "nodes": [
            {"name": "test-node-1", "primary": "http://10.0.0.1:8001/v1"}
        ]
    }
    config_file.write_text(yaml.dump(initial_content))
    monkeypatch.setenv("CONFIG_PATH", str(config_file))

    state.CONFIG = state.load_config()
    assert len(state.CONFIG["nodes"]) == 1

    assert state.reload_config_if_changed() is False

    updated_content = {
        "nodes": [
            {"name": "test-node-1", "primary": "http://10.0.0.1:8001/v1"},
            {"name": "test-node-2", "primary": "http://10.0.0.2:8000/v1"}
        ]
    }
    os.utime(str(config_file), (time.time() + 5, time.time() + 5))
    config_file.write_text(yaml.dump(updated_content))

    assert state.reload_config_if_changed() is True
    assert len(state.CONFIG["nodes"]) == 2
    assert "test-node-2" in state.ACTIVE_REQUESTS


async def test_manual_reload_endpoint(tmp_path, monkeypatch):
    """Test manual_reload_config management endpoint."""
    config_file = tmp_path / "config.yaml"
    initial_content = {"nodes": [{"name": "test-node-1", "primary": "http://10.0.0.1:8001/v1"}]}
    config_file.write_text(yaml.dump(initial_content))
    monkeypatch.setenv("CONFIG_PATH", str(config_file))

    state.CONFIG = state.load_config()

    res = await app_module.manual_reload_config()
    assert res["status"] == "unchanged"
    assert res["nodes_count"] == 1
    assert res["nodes"] == ["test-node-1"]
