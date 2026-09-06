# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import re
from typing import List

from router import state
from router.logging_setup import logger


def normalize_model_str(name: str) -> str:
    """Normalizes model string by removing organization prefix, slashes, hyphens, underscores, and dots."""
    if not name:
        return ""
    s = str(name).strip().lower()
    if "/" in s:
        s = s.split("/")[-1]
    return re.sub(r'[^a-z0-9]', '', s)


def is_model_matching(requested: str, served: str) -> bool:
    """Returns True if requested model matches served model via exact, case-insensitive, alias, or fuzzy normalized matching."""
    if not requested or not served:
        return False
    if requested == served:
        return True
    if requested.lower() == served.lower():
        return True
    aliases = state.CONFIG.get("model_aliases", {})
    target_alias = aliases.get(requested) or aliases.get(requested.lower())
    if target_alias and (target_alias == served or target_alias.lower() == served.lower()):
        return True
    norm_req = normalize_model_str(requested)
    norm_srv = normalize_model_str(served)
    if norm_req == norm_srv:
        return True
    for alias_k, alias_v in aliases.items():
        if normalize_model_str(alias_k) == norm_req and normalize_model_str(alias_v) == norm_srv:
            return True
    return False


def get_canonical_model_name(requested_model: str, active_models: List[str]) -> str:
    """Resolves requested model string to the canonical model name served by active cluster nodes."""
    if not requested_model:
        return requested_model
    for active in active_models:
        if is_model_matching(requested_model, active):
            return active
    aliases = state.CONFIG.get("model_aliases", {})
    if requested_model in aliases:
        return aliases[requested_model]
    if requested_model.lower() in aliases:
        return aliases[requested_model.lower()]
    return requested_model


def get_model_capabilities(model_name: str) -> dict:
    """Resolves the capabilities dictionary for a given model from the config catalog.
    If no pattern matches, defaults to all capabilities being True.
    """
    model_name_lower = model_name.lower()
    catalog = state.CONFIG.get("model_capabilities", [])

    for entry in catalog:
        pattern = entry.get("name_pattern", "").lower()
        if pattern and pattern in model_name_lower:
            return {
                "vision": entry.get("vision", True),
                "tool_calling": entry.get("tool_calling", True),
                "structured_output": entry.get("structured_output", True),
            }

    return {
        "vision": True,
        "tool_calling": True,
        "structured_output": True,
    }


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


def check_and_reroute_capabilities(json_data: dict) -> bool:
    """Detects if a model was requested with requirements it does not support,
    and rewrites the model parameter to a suitable active fallback model.
    Returns True if the model was rewritten.
    """
    routing_cfg = state.CONFIG.get("capabilities_routing", {})
    if not routing_cfg.get("enabled", True):
        return False

    needs_vision = has_image_content(json_data)
    needs_tool_calling = "tools" in json_data or "functions" in json_data

    needs_structured = False
    resp_format = json_data.get("response_format")
    if isinstance(resp_format, dict):
        if resp_format.get("type") in ("json_object", "json_schema"):
            needs_structured = True

    if not (needs_vision or needs_tool_calling or needs_structured):
        return False

    requested_model = json_data.get("model", "")
    req_caps = get_model_capabilities(requested_model)

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

    all_active_models = []
    for models in state.NODE_MODELS_CACHE.values():
        for m in models:
            if m not in all_active_models:
                all_active_models.append(m)

    target_model = None
    for active_model in all_active_models:
        caps = get_model_capabilities(active_model)

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
