# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Thermal-aware routing: temperature polling plus the two ALWAYS-ON
pipeline stages (critical hard-block, prefix-affinity cooling override)
that run in router/app.py's handle_llm_request before any routing
strategy sees the node list. These aren't strategy-specific -- every
strategy operates on whatever eligible_nodes survives Stage 1, and
Stage 2 can hand back an already-decided node before a strategy even
runs. filter_cool_nodes() is a third, opt-in helper any strategy can
call (only "smart" does today).
"""
from typing import Dict, List, Optional

from router import metrics, model_matching, state
from router.logging_setup import logger


async def fetch_node_temperature(node: Dict[str, str]) -> Optional[float]:
    """Attempts to fetch node hardware/GPU temperature via node-exporter on port 9100.
    Returns maximum temperature in Celsius or None if telemetry/exporter is unavailable.
    """
    thermal_cfg = state.CONFIG.get("thermal_routing", {})
    if not thermal_cfg.get("enabled", True):
        return None

    primary_url = node.get("primary", "")
    try:
        host = primary_url.split("://")[-1].split(":")[0]
    except Exception:
        return None

    port = thermal_cfg.get("exporter_port", 9100)
    metrics_url = f"http://{host}:{port}/metrics"

    try:
        resp = await state.CLIENT.get(metrics_url, timeout=1.5)
        if resp.status_code == 200:
            temps = []
            for line in resp.text.splitlines():
                if line.startswith("node_hwmon_temp_celsius{") or line.startswith("node_thermal_zone_temp{") or line.startswith("DCGM_FI_DEV_GPU_TEMP") or line.startswith("nvidia_smi_temperature_celsius"):
                    # Ignore storage NVMe, Wi-Fi, and NIC PHY sensors so routing isolates GPU/SoC core heat
                    line_lower = line.lower()
                    if "nvme" in line_lower or "wifi" in line_lower or "phy" in line_lower:
                        continue
                    try:
                        val = float(line.split()[-1])
                        # Filter out invalid sensor sentinel values (e.g. >150 or <=0)
                        if 0 < val < 150:
                            temps.append(val)
                    except ValueError:
                        continue
            if temps:
                return max(temps)
    except Exception:
        pass
    return None


def apply_stage1_critical_filter(eligible_nodes: List[dict], requested_model: str) -> List[dict]:
    """STAGE 1: Critical Thermal Hard Block. Excludes nodes >= critical_temp
    (default 80C) from the eligible pool entirely -- this runs before any
    routing strategy sees the node list, regardless of which strategy is
    configured. Returns the original list unchanged if thermal routing is
    disabled, or if excluding hot nodes would leave nothing eligible
    (better to route somewhere than nowhere; the reroute-to-coolest
    fallback further down handles that case)."""
    thermal_cfg = state.CONFIG.get("thermal_routing", {})
    if not thermal_cfg.get("enabled", True):
        return eligible_nodes

    critical_temp = float(thermal_cfg.get("critical_temp_celsius", 80.0))
    cool_and_warm_nodes = [n for n in eligible_nodes if state.NODE_TEMP_CACHE.get(n["name"], 0) < critical_temp]
    if cool_and_warm_nodes:
        return cool_and_warm_nodes

    logger.warning(f"All eligible nodes for '{requested_model}' exceed critical thermal limit ({critical_temp}°C). Routing to coolest available.")
    if metrics.HAS_PROMETHEUS:
        metrics.ROUTER_THERMAL_CRITICAL_HARD_BLOCKS.labels(node="all_critical").inc()
        metrics.ROUTER_THERMAL_EVENTS.labels(event_type="Critical Hard Block (80°C)").inc()
    return eligible_nodes


def apply_stage2_prefix_affinity(
    cached_node_name: Optional[str],
    eligible_nodes: List[dict],
    requested_model: str,
    est_request_len: int,
) -> Optional[dict]:
    """STAGE 2: if a prefix-hash match named a node that's still eligible,
    normally route back to it to keep its KV cache warm -- UNLESS any of
    three conditions holds, in which case the cache hit is deliberately
    bypassed and the caller falls through to whichever routing strategy
    is configured:

    1. Thermal: the node is at/above the warning threshold (original
       behavior, protects hardware).
    2. Context: the node's reported context window is smaller than this
       request's estimated length -- closes a real gap where the
       affinity shortcut used to skip the context check the "smart"
       strategy's own fallback path already does.
    3. Load imbalance: the node is running significantly more concurrent
       requests than another eligible node right now
       (thermal_routing.max_affinity_load_imbalance, default 3; 0
       disables this check). Added after live data showed one
       consistently-cool node (never once tripping the thermal check)
       absorbing a hugely disproportionate share of traffic purely
       because it never got the chance to cool down and get bypassed --
       this makes "significantly worse than an available alternative
       right now" its own explicit bypass reason, independent of heat.

    Returns the matched node dict, or None if there was no match / it
    was bypassed for any of the above."""
    if not cached_node_name:
        return None

    thermal_cfg = state.CONFIG.get("thermal_routing", {})
    thermal_enabled = thermal_cfg.get("enabled", True)
    warning_temp = float(thermal_cfg.get("warning_temp_celsius", 70.0))
    thermal_over_kv = thermal_cfg.get("thermal_priority_over_kv_cache", True)
    max_imbalance = thermal_cfg.get("max_affinity_load_imbalance", 3)

    for n in eligible_nodes:
        if n["name"] != cached_node_name:
            continue

        node_temp = state.NODE_TEMP_CACHE.get(n["name"], 0)
        if thermal_enabled and thermal_over_kv and node_temp >= warning_temp:
            logger.info(f"Bypassing KV-cache hit for node '{n['name']}' because temp ({node_temp:.1f}°C) exceeds warning threshold ({warning_temp}°C) to allow cooling.")
            if metrics.HAS_PROMETHEUS:
                metrics.ROUTER_KV_CACHE_BYPASS_COOLING.labels(node=n['name']).inc()
                metrics.ROUTER_THERMAL_EVENTS.labels(event_type="KV-Cache Bypass (70°C)").inc()
            return None

        node_ctx = model_matching.node_context_window(n["name"], requested_model)
        if node_ctx > 0 and est_request_len > node_ctx:
            logger.info(f"Bypassing KV-cache hit for node '{n['name']}': est request len ({est_request_len}) exceeds its context limit ({node_ctx}).")
            return None

        if max_imbalance > 0:
            matched_active = state.ACTIVE_REQUESTS.get(n["name"], 0)
            other_actives = [state.ACTIVE_REQUESTS.get(o["name"], 0) for o in eligible_nodes if o["name"] != n["name"]]
            if other_actives and (matched_active - min(other_actives)) >= max_imbalance:
                logger.info(f"Bypassing KV-cache hit for node '{n['name']}': {matched_active} active requests vs {min(other_actives)} on the least-loaded eligible node (imbalance >= {max_imbalance}).")
                if metrics.HAS_PROMETHEUS:
                    metrics.ROUTER_KV_CACHE_BYPASS_LOAD_IMBALANCE.labels(node=n['name']).inc()
                    metrics.ROUTER_THERMAL_EVENTS.labels(event_type="KV-Cache Bypass (Load Imbalance)").inc()
                return None

        return n
    return None


def filter_cool_nodes(candidate_nodes: List[dict], requested_model: str) -> List[dict]:
    """Filters candidate_nodes down to those below the warning threshold,
    incrementing the coolest-node-reroute counters if that would leave
    nothing (in which case the original, unfiltered list is returned so a
    strategy still has somewhere to route). No-op if thermal routing is
    disabled. Used by the "smart" strategy today; available to any
    strategy that wants the same behavior."""
    thermal_cfg = state.CONFIG.get("thermal_routing", {})
    if not thermal_cfg.get("enabled", True):
        return candidate_nodes

    warning_temp = float(thermal_cfg.get("warning_temp_celsius", 70.0))
    cool_nodes = [n for n in candidate_nodes if state.NODE_TEMP_CACHE.get(n["name"], 0) < warning_temp]
    if cool_nodes:
        return cool_nodes

    logger.warning(f"All candidate nodes for '{requested_model}' exceed warning temp ({warning_temp}°C). Routing to coolest available.")
    if metrics.HAS_PROMETHEUS:
        metrics.ROUTER_THERMAL_REROUTES_COOLEST.labels(node="all_warning").inc()
        metrics.ROUTER_THERMAL_EVENTS.labels(event_type="Coolest-Node Reroute").inc()
    return candidate_nodes
