#!/usr/bin/env python3
"""
LLM Cluster Router Test Suite
Tests tool calling, streaming, models listing, and context windows across cluster routers.
Usage: python3 test_routers.py
"""

import sys
import json
import urllib.request
import urllib.error
import ssl

DOMAINS = [
    "https://llm.rkdfv.com",
    "https://qwen.rkdfv.com",
    "https://llm.example.com",
    "https://qwen.example.com",
    "https://llm.example.com",
    "https://qwen.example.com"
]

# Disable SSL verification for local testing if self-signed certs are used
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def test_models_endpoint(base_url):
    url = f"{base_url}/v1/models"
    print(f"\n[1] Testing Models Listing: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LLM-Test-Client/1.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            print(f"    Status: {resp.status} OK")
            print(f"    Active Models ({len(models)}):")
            for m in models:
                m_id = m.get("id")
                ctx_len = m.get("max_model_len", m.get("context_window", "N/A"))
                print(f"      - {m_id} (Context Window: {ctx_len})")
            return models[0]["id"] if models else None
    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def test_tool_calling(base_url, model_name):
    url = f"{base_url}/v1/chat/completions"
    print(f"\n[2] Testing Tool Calling (Function Calling): {url}")
    payload = {
        "model": model_name or "Qwen/Qwen3.8-27B-FP8",
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": "auto"
    }

    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LLM-Test-Client/1.0"
            }
        )
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            choice = res["choices"][0]
            msg = choice.get("message", {})
            tool_calls = msg.get("tool_calls", [])
            finish_reason = choice.get("finish_reason")

            print(f"    Status: {resp.status} OK")
            print(f"    Finish Reason: {finish_reason}")
            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                print(f"    SUCCESS: Generated Tool Call -> Name: '{func.get('name')}', Args: {func.get('arguments')}")
            else:
                print(f"    Response Content: {msg.get('content')}")
    except Exception as e:
        print(f"    FAILED: {e}")


def test_streaming(base_url, model_name):
    url = f"{base_url}/v1/chat/completions"
    print(f"\n[3] Testing SSE Streaming Response: {url}")
    payload = {
        "model": model_name or "Qwen/Qwen3.8-27B-FP8",
        "messages": [
            {"role": "user", "content": "Count from 1 to 3."}
        ],
        "stream": True,
        "max_tokens": 20
    }

    try:
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LLM-Test-Client/1.0"
            }
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            print(f"    Status: {resp.status} OK")
            chunk_count = 0
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: ") and line_str != "data: [DONE]":
                    chunk_count += 1
            print(f"    SUCCESS: Received {chunk_count} SSE streaming chunks")
    except Exception as e:
        print(f"    FAILED: {e}")


def main():
    print("==================================================")
    print("      LLM Cluster Router End-to-End Test Suite    ")
    print("==================================================")

    for domain in DOMAINS:
        print(f"\n==================================================")
        print(f" Target Domain: {domain}")
        print(f"==================================================")
        model = test_models_endpoint(domain)
        if model:
            test_tool_calling(domain, model)
            test_streaming(domain, model)


if __name__ == "__main__":
    main()
