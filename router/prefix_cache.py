# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Prefix-hash -> last-used-node affinity tracking.

`record()` is called from exactly ONE place in the request pipeline
(router/app.py, right after a node has been selected -- by prefix-affinity
OR by any routing strategy), unconditionally. That single call site is
the fix for a real bug this refactor closes off: the previous inline
version only recorded a PREFIX_CACHE entry from the "smart" strategy's
code path, because a stray `if selected_node:` block sat between the
smart-mode branch and the sticky/random branches, silently absorbing the
`elif`/`else` that were meant to be part of the *strategy* selection, not
the cache-recording step. Under `routing.mode: sticky` or the bare random
fallback, PREFIX_CACHE was never seeded at all -- prefix-affinity routing
would have silently never activated for those modes. Structuring this as
"select a node (however), then always record" makes that class of bug
impossible: there is exactly one place that writes PREFIX_CACHE, and it
runs no matter which strategy produced the selection.
"""
import time
import zlib
from typing import Dict, Optional, Tuple

from router import state

# {prefix_hash: (node_name, timestamp)}
PREFIX_CACHE: Dict[str, Tuple[str, float]] = {}


def get_prefix_hash(json_data: dict) -> str:
    """Extracts system prompt or initial prompt prefix and returns a CRC32
    hash string. Returns empty string if prefix length is below
    min_prefix_length or prefix-cache routing is disabled.
    """
    prefix_cfg = state.CONFIG.get("prefix_cache_routing", {})
    if not prefix_cfg.get("enabled", True):
        return ""

    min_len = prefix_cfg.get("min_prefix_length", 50)
    messages = json_data.get("messages")
    prefix_text = ""

    if isinstance(messages, list) and messages:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                prefix_text = str(msg.get("content", ""))
                break
        if not prefix_text and isinstance(messages[0], dict):
            prefix_text = str(messages[0].get("content", ""))
    elif "prompt" in json_data:
        prefix_text = str(json_data.get("prompt", ""))

    if len(prefix_text) >= min_len:
        return f"{zlib.crc32(prefix_text.encode('utf-8')):x}"

    return ""


async def lookup(prefix_hash: str) -> Optional[str]:
    """Returns the node name last used for this prefix hash, if any and
    still within prefix_cache_routing.ttl_seconds -- else None (and
    evicts the stale entry)."""
    if not prefix_hash:
        return None
    prefix_cfg = state.CONFIG.get("prefix_cache_routing", {})
    ttl = prefix_cfg.get("ttl_seconds", 600)
    now = time.time()
    async with state.CACHE_LOCK:
        if prefix_hash in PREFIX_CACHE:
            node_name, recorded_at = PREFIX_CACHE[prefix_hash]
            if now - recorded_at < ttl:
                return node_name
            del PREFIX_CACHE[prefix_hash]
    return None


async def record(prefix_hash: str, node_name: str) -> None:
    """Records that `node_name` handled `prefix_hash`, for future lookup()
    calls. Safe to call unconditionally -- callers should call this after
    every successful node selection when prefix_hash is non-empty."""
    async with state.CACHE_LOCK:
        PREFIX_CACHE[prefix_hash] = (node_name, time.time())
