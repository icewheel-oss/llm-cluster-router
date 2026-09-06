# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import json

from router.logging_setup import logger


def sanitize_tools(json_data: dict) -> bool:
    """Sanitizes OpenAI tool/function definitions to prevent vLLM/Jinja template type errors.
    Returns True if modifications were made.
    """
    modified = False

    tools = json_data.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if "function" in tool:
                func = tool["function"]
                if not isinstance(func, dict):
                    continue

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

                if "properties" not in parameters or not isinstance(parameters.get("properties"), dict) or parameters["properties"] is None:
                    parameters["properties"] = {}
                    modified = True

    functions = json_data.get("functions")
    if isinstance(functions, list):
        for func in functions:
            if not isinstance(func, dict):
                continue

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
