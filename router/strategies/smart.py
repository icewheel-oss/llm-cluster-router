# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import zlib

from router import state, thermal
from router.logging_setup import logger
from router.strategies import RoutingContext, find_header_case_insensitive, register_strategy


@register_strategy("smart")
def select_node(ctx: RoutingContext) -> dict:
    """If the caller sent a sticky-session header, hash-route on it (same
    mechanism as the "sticky" strategy, but opt-in per-request rather than
    always falling back to auth-user/client-IP). Otherwise picks the
    least-loaded node among those that fit the request's estimated
    context length and aren't running warm (thermal-aware)."""
    sticky_header = ctx.routing_config.get("sticky_header", "x-session-id").lower()
    sticky_key = find_header_case_insensitive(ctx.headers, sticky_header)

    if sticky_key:
        sorted_nodes = sorted(ctx.eligible_nodes, key=lambda n: n["name"])
        hash_val = zlib.crc32(sticky_key.encode("utf-8"))
        selected_node = sorted_nodes[hash_val % len(sorted_nodes)]
        logger.info(f"Smart routing selected sticky node {selected_node['name']} for key '{sticky_key}'")
        return selected_node

    # Context Window & Thermal-aware routing
    # Estimate requested token length (approx 3.2 chars per token for safe estimation + max_tokens requested)
    est_prompt_tokens = int(len(ctx.prompt) / 3.2) if ctx.prompt else 0
    max_gen_tokens = ctx.json_data.get("max_tokens", 2048) if isinstance(ctx.json_data, dict) else 2048
    est_total_request_len = est_prompt_tokens + max_gen_tokens

    candidate_nodes = list(ctx.eligible_nodes)

    # Filter nodes by context window capacity if max_model_len reported
    context_capable_nodes = []
    for n in candidate_nodes:
        node_models = state.NODE_MODELS_CACHE.get(n["name"], [])
        node_ctx = 0
        for m in node_models:
            m_id = m.get("id") if isinstance(m, dict) else str(m)
            if m_id == ctx.requested_model:
                node_ctx = m.get("max_model_len", m.get("context_window", 0)) if isinstance(m, dict) else 0
                break
        # If node reports max_model_len and it's smaller than estimated request, skip node
        if node_ctx > 0 and est_total_request_len > node_ctx:
            logger.info(f"Skipping node '{n['name']}' for '{ctx.requested_model}': est request len ({est_total_request_len}) exceeds node context limit ({node_ctx})")
            continue
        context_capable_nodes.append(n)

    if context_capable_nodes:
        candidate_nodes = context_capable_nodes
    else:
        logger.warning(f"No active node for '{ctx.requested_model}' supports est context length ({est_total_request_len}). Routing to all eligible nodes.")

    candidate_nodes = thermal.filter_cool_nodes(candidate_nodes, ctx.requested_model)

    # Sort by active requests, then node priority (lower = higher priority), then lower temperature
    selected_node = min(
        candidate_nodes,
        key=lambda n: (
            state.ACTIVE_REQUESTS.get(n["name"], 0),
            n.get("priority", 1),
            state.NODE_TEMP_CACHE.get(n["name"], 0)
        )
    )
    active_count = state.ACTIVE_REQUESTS.get(selected_node["name"], 0)
    node_temp = state.NODE_TEMP_CACHE.get(selected_node["name"])
    temp_info = f", temp: {node_temp:.1f}°C" if node_temp is not None else ""
    logger.info(f"Smart routing selected node {selected_node['name']} (active: {active_count}{temp_info})")
    return selected_node
