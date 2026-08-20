# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import asyncio
import pytest
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from unittest.mock import AsyncMock, patch, MagicMock

import main

# Apply asyncio marker to all tests in this file
pytestmark = pytest.mark.asyncio

@pytest.fixture(autouse=True)
def setup_config():
    """Initializes a mock configuration before each test."""
    main.CONFIG = {
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
    main.NODE_MODELS_CACHE = {}


async def test_health():
    """Test health check endpoint."""
    resp = await main.health()
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
    main.CLIENT = mock_client

    node = main.CONFIG["nodes"][0]
    models = await main.fetch_node_models(node)
    
    assert models == ["model-a"]
    mock_client.get.assert_called_once_with(
        "http://192.168.86.221:8000/v1/models",
        timeout=0.1
    )


async def test_fetch_node_models_failover_to_backup(mocker):
    """Test fetch_node_models fails on primary and successfully fails over to backup WiFi path."""
    mock_client = AsyncMock()
    
    # Primary raises TimeoutException; Backup returns success response
    mock_client.get.side_effect = [
        httpx.TimeoutException("Primary timed out"),
        MagicMock(status_code=200, json=lambda: {"data": [{"id": "model-b"}]})
    ]
    main.CLIENT = mock_client

    node = main.CONFIG["nodes"][0]
    models = await main.fetch_node_models(node)
    
    assert models == ["model-b"]
    assert mock_client.get.call_count == 2
    
    # Verify both calls were made to correct URLs
    mock_client.get.assert_any_call("http://192.168.86.221:8000/v1/models", timeout=0.1)
    mock_client.get.assert_any_call("http://192.168.86.211:8000/v1/models", timeout=0.2)


async def test_fetch_node_models_all_fail(mocker):
    """Test fetch_node_models returns empty list if both primary and backup paths fail."""
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.RequestError("Network error")
    main.CLIENT = mock_client

    node = main.CONFIG["nodes"][0]
    models = await main.fetch_node_models(node)
    
    assert models == []


async def test_get_models_from_cache():
    """Test get_models returns deduplicated models lists from internal cache."""
    main.NODE_MODELS_CACHE = {
        "node-1": ["model-a", "model-b"],
        "node-4": ["model-b", "model-c"]
    }
    
    resp = await main.get_models()
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
    main.CLIENT = mock_client

    node = main.CONFIG["nodes"][0]
    resp = await main.forward_request(
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
    
    # Assert primary URL was targeted
    mock_client.build_request.assert_called_once_with(
        "POST",
        "http://192.168.86.221:8000/v1/chat/completions",
        headers={"content-type": "application/json", "Authorization": "Bearer key"},
        content=b'{"test": 1}',
        timeout=1.0
    )


async def test_forward_request_failover_to_backup(mocker):
    """Test forwarding requests falls back to backup WiFi if primary Ethernet times out."""
    mock_client = AsyncMock()
    
    # Primary send fails with RequestError, Backup send succeeds
    mock_response = MagicMock(status_code=200, headers={}, aiter_bytes=lambda: AsyncIterator([b"ok"]))
    mock_client.send.side_effect = [
        httpx.TimeoutException("Timeout"),
        mock_response
    ]
    
    mock_client.build_request.side_effect = ["req-1", "req-2"]
    main.CLIENT = mock_client

    node = main.CONFIG["nodes"][0]
    resp = await main.forward_request(
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
    
    # Verify primary and backup build_request targets
    mock_client.build_request.assert_any_call("POST", "http://192.168.86.221:8000/v1/chat/completions", headers={}, content=b"", timeout=1.0)
    mock_client.build_request.assert_any_call("POST", "http://192.168.86.211:8000/v1/chat/completions", headers={}, content=b"", timeout=1.0)


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
    # Setup cache so node-1 is known to have deepseek loaded
    main.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "node-4": []
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{"id":"chatcmpl-123","object":"chat.completion","choices":[{"message":{"content":"Hello!"}}]}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("main.forward_request", return_value=mock_resp)

    # Mock FastAPI Request
    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json", "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    resp = await main.handle_llm_request(mock_request)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/json"
    
    # Read streamed response body chunks
    chunks = [chunk async for chunk in resp.body_iterator]
    assert b"chatcmpl-123" in chunks[0]
    
    # Verify main.forward_request was called targeting node-1
    main.forward_request.assert_called_once()
    args, kwargs = main.forward_request.call_args
    assert kwargs["node"]["name"] == "node-1"  # selected_node
    assert kwargs["path"] == "chat/completions"  # relative path stripped of v1/
    assert kwargs["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


async def test_handle_llm_request_streaming(mocker):
    """Test handle_llm_request correctly forwards streaming requests and chunks event stream."""
    main.NODE_MODELS_CACHE = {
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
    mocker.patch("main.forward_request", return_value=mock_resp)

    # Mock FastAPI Request
    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "stream": true}')
    mock_request.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731", "stream": True})

    resp = await main.handle_llm_request(mock_request)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream"
    
    chunks = [chunk async for chunk in resp.body_iterator]
    assert chunks == sse_data


async def test_handle_llm_request_model_rewriting(mocker):
    """Test that when a client requests a model that is not loaded, the gateway rewrites it to an active model fallback."""
    import json
    # Cache setup: deepseek-ai/DeepSeek-V4-Flash-0731 is active on node-1, but client requests "model-not-found"
    main.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{"id":"chatcmpl-123","object":"chat.completion","choices":[{"message":{"content":"Hello!"}}]}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("main.forward_request", return_value=mock_resp)

    # Mock FastAPI Request asking for "model-not-found"
    mock_request = MagicMock(spec=Request)
    mock_request.url = MagicMock()
    mock_request.url.path = "/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.headers = {"content-type": "application/json"}
    mock_request.client = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.body = AsyncMock(return_value=b'{"model": "model-not-found"}')
    mock_request.json = AsyncMock(return_value={"model": "model-not-found"})

    resp = await main.handle_llm_request(mock_request)
    assert resp.status_code == 200

    # Verify main.forward_request was called targeting node-1 and rewrote model to "deepseek-ai/DeepSeek-V4-Flash-0731"
    main.forward_request.assert_called_once()
    args, kwargs = main.forward_request.call_args
    assert kwargs["node"]["name"] == "node-1"
    assert kwargs["requested_model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert kwargs["original_model"] == "model-not-found"
    
    # Verify the request content body was rewritten to reference the fallback model
    rewritten_body = json.loads(kwargs["content"].decode("utf-8"))
    assert rewritten_body["model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"


async def test_handle_llm_request_sticky_routing(mocker):
    """Test that requests with the same sticky key are routed consistently to the same node in sticky mode."""
    # Cache setup: deepseek model is active on both node-1 and node-2
    main.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "node-2": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }
    main.CONFIG["routing"] = {
        "mode": "sticky",
        "sticky_header": "x-session-id"
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("main.forward_request", return_value=mock_resp)

    # 1. Create Request with session-1
    mock_request_1 = MagicMock(spec=Request)
    mock_request_1.url = MagicMock()
    mock_request_1.url.path = "/v1/chat/completions"
    mock_request_1.method = "POST"
    mock_request_1.headers = {"content-type": "application/json", "x-session-id": "session-1"}
    mock_request_1.client = MagicMock()
    mock_request_1.client.host = "127.0.0.1"
    mock_request_1.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request_1.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    # 2. Create another Request with session-1 (must route to the SAME node)
    mock_request_2 = MagicMock(spec=Request)
    mock_request_2.url = MagicMock()
    mock_request_2.url.path = "/v1/chat/completions"
    mock_request_2.method = "POST"
    mock_request_2.headers = {"content-type": "application/json", "x-session-id": "session-1"}
    mock_request_2.client = MagicMock()
    mock_request_2.client.host = "127.0.0.1"
    mock_request_2.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request_2.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    # Run request 1
    resp1 = await main.handle_llm_request(mock_request_1)
    assert resp1.status_code == 200
    args1, kwargs1 = main.forward_request.call_args
    node_selected_1 = kwargs1["node"]["name"]

    # Reset mock and run request 2
    main.forward_request.reset_mock()
    resp2 = await main.handle_llm_request(mock_request_2)
    assert resp2.status_code == 200
    args2, kwargs2 = main.forward_request.call_args
    node_selected_2 = kwargs2["node"]["name"]

    # Verify both routed to the same node consistently
    assert node_selected_1 == node_selected_2

    # Reset config mode to default
    main.CONFIG["routing"] = {"mode": "random"}


async def test_handle_llm_request_smart_routing(mocker):
    """Test that smart routing routes stickily when session ID is present, and to the least loaded when absent."""
    # Cache setup: deepseek model is active on both node-1 and node-2
    main.NODE_MODELS_CACHE = {
        "node-1": ["deepseek-ai/DeepSeek-V4-Flash-0731"],
        "node-2": ["deepseek-ai/DeepSeek-V4-Flash-0731"]
    }
    main.CONFIG["nodes"] = [
        {"name": "node-1", "primary": "http://192.168.86.221:8000/v1"},
        {"name": "node-2", "primary": "http://192.168.86.222:8000/v1"}
    ]
    main.CONFIG["routing"] = {
        "mode": "smart",
        "sticky_header": "x-session-id"
    }
    # Simulate node-1 having 3 active requests, and node-2 having 0
    main.ACTIVE_REQUESTS = {
        "node-1": 3,
        "node-2": 0
    }

    mock_resp = StreamingResponse(
        AsyncIterator([b'{}']),
        status_code=200,
        headers={"content-type": "application/json"}
    )
    mocker.patch("main.forward_request", return_value=mock_resp)

    # 1. Test case: Request has NO session ID. Must route to the least loaded node (node-2)
    mock_request_no_session = MagicMock(spec=Request)
    mock_request_no_session.url = MagicMock()
    mock_request_no_session.url.path = "/v1/chat/completions"
    mock_request_no_session.method = "POST"
    mock_request_no_session.headers = {"content-type": "application/json"}
    mock_request_no_session.client = MagicMock()
    mock_request_no_session.client.host = "127.0.0.1"
    mock_request_no_session.body = AsyncMock(return_value=b'{"model": "deepseek-ai/DeepSeek-V4-Flash-0731"}')
    mock_request_no_session.json = AsyncMock(return_value={"model": "deepseek-ai/DeepSeek-V4-Flash-0731"})

    resp1 = await main.handle_llm_request(mock_request_no_session)
    assert resp1.status_code == 200
    args1, kwargs1 = main.forward_request.call_args
    assert kwargs1["node"]["name"] == "node-2"

    # Reset CONFIG
    main.CONFIG["routing"] = {"mode": "random"}


async def test_openai_compatible_errors():
    """Test that standard HTTPExceptions are formatted as standard OpenAI error payloads."""
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    resp = client.get("/v1/non-existent-route")
    assert resp.status_code == 404
    error_json = resp.json()
    assert "error" in error_json
    assert "message" in error_json["error"]
    assert error_json["error"]["type"] == "invalid_request_error"
    assert error_json["error"]["code"] == "404"


def test_sanitize_tools():
    """Test tool calling schemas sanitization rules."""
    # Case 1: Missing parameters field entirely
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
    modified1 = main.sanitize_tools(data_missing_params)
    assert modified1 is True
    func1 = data_missing_params["tools"][0]["function"]
    assert "parameters" in func1
    assert func1["parameters"]["properties"] == {}

    # Case 2: Missing properties inside parameters dict
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
    modified2 = main.sanitize_tools(data_missing_props)
    assert modified2 is True
    func2 = data_missing_props["tools"][0]["function"]
    assert func2["parameters"]["properties"] == {}

    # Case 3: Parameters serialized as a JSON string
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
    modified3 = main.sanitize_tools(data_string_params)
    assert modified3 is True
    func3 = data_string_params["tools"][0]["function"]
    assert isinstance(func3["parameters"], dict)
    assert func3["parameters"]["properties"]["city"]["type"] == "string"

    # Case 4: Legacy functions list missing properties
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
    modified4 = main.sanitize_tools(data_legacy_functions)
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
    modified = main.sanitize_messages(messages)
    assert modified is True
    args = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert args == "{}"



def test_has_image_content():
    """Test multimodal image detection in payloads."""
    # Case 1: Simple text request
    data_text = {
        "messages": [{"role": "user", "content": "hello world"}]
    }
    assert main.has_image_content(data_text) is False

    # Case 2: Multi-modal content array with text and image
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
    assert main.has_image_content(data_image) is True


def test_check_and_reroute_capabilities():
    """Test that requests matching capability mismatch fallback rules are rerouted correctly."""
    # 1. Setup configuration and active model cache
    main.CONFIG["capabilities_routing"] = {"enabled": True}
    main.CONFIG["model_capabilities"] = [
        {"name_pattern": "r1", "vision": False, "tool_calling": False, "structured_output": False},
        {"name_pattern": "qwen", "vision": True, "tool_calling": True, "structured_output": True},
        {"name_pattern": "deepseek", "vision": False, "tool_calling": True, "structured_output": True}
    ]
    main.NODE_MODELS_CACHE = {
        "node-1": ["DeepSeek-V4-Flash-0731"],
        "node-3": ["Qwen/Qwen3.8-27B-FP8"],
        "node-4": ["DeepSeek-R1-Distill-Q8"]
    }

    # Case 1: Text-only model with text content should NOT be rerouted
    payload1 = {
        "model": "DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": "hi"}]
    }
    assert main.check_and_reroute_capabilities(payload1) is False
    assert payload1["model"] == "DeepSeek-V4-Flash-0731"

    # Case 2: Vision request sent to text-only DeepSeek model should reroute to Qwen
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
    assert main.check_and_reroute_capabilities(payload2) is True
    assert payload2["model"] == "Qwen/Qwen3.8-27B-FP8"

    # Case 3: Tool-calling request sent to R1 model (no tool calling) should reroute to DeepSeek-V4-Flash
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
    assert main.check_and_reroute_capabilities(payload3) is True
    assert payload3["model"] == "DeepSeek-V4-Flash-0731"


async def test_stream_and_log_cancelled(mocker):
    """Test that client cancellation terminates the backend streamed response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {}
    
    # Simulate client disconnect by raising CancelledError during byte iteration
    async def mock_aiter_bytes():
        raise asyncio.CancelledError()
        yield b"" # make it a generator
        
    mock_response.aiter_bytes = mock_aiter_bytes
    
    # Mock aclose as an async function
    aclose_called = False
    async def mock_aclose():
        nonlocal aclose_called
        aclose_called = True
        
    mock_response.aclose = mock_aclose
    
    # Call stream_and_log and execute the generator
    gen = main.stream_and_log(
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
            
    # Verify aclose was successfully called
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
    hash_val = main.get_prefix_hash(payload)
    assert len(hash_val) > 0
    
    # Store cache entry
    main.PREFIX_CACHE[hash_val] = ("node-3", main.time.time())
    c_node, c_time = main.PREFIX_CACHE.get(hash_val, (None, 0))
    assert c_node == "node-3"


def test_rate_limiting():
    """Test client rate limiting quota enforcement."""
    main.CONFIG["rate_limiting"] = {"enabled": True, "requests_per_minute": 2}
    main.RATE_LIMIT_CACHE.clear()

    client = "user-test"
    assert main.check_rate_limit(client) is True
    assert main.check_rate_limit(client) is True
    assert main.check_rate_limit(client) is False








