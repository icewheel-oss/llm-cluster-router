# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Always session-affine, unlike "smart" (which only goes sticky when a
session header is actually present). Useful when every request should
land on a consistent node for a given caller, even with no explicit
session header, by falling back to auth-user then client-IP."""
import zlib

from router.logging_setup import logger
from router.strategies import RoutingContext, find_header_case_insensitive, register_strategy


@register_strategy("sticky")
def select_node(ctx: RoutingContext) -> dict:
    """Hashes a sticky key (session header, else auth user, else client IP)
    to consistently pick the same node across requests from the same
    caller -- session affinity, independent of prefix-cache affinity."""
    sticky_header = ctx.routing_config.get("sticky_header", "x-session-id").lower()
    sticky_key = find_header_case_insensitive(ctx.headers, sticky_header)
    if not sticky_key:
        sticky_key = ctx.auth_user if ctx.auth_user != "anonymous" else ctx.client_ip

    sorted_nodes = sorted(ctx.eligible_nodes, key=lambda n: n["name"])
    hash_val = zlib.crc32(sticky_key.encode("utf-8"))
    selected_node = sorted_nodes[hash_val % len(sorted_nodes)]
    logger.info(f"Sticky routing selected node {selected_node['name']} for key '{sticky_key}'")
    return selected_node
