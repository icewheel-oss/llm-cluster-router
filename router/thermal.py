# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

from typing import Dict, List, Optional

from router import metrics, state
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


def apply_stage2_prefix_affinity(cached_node_name: Optional[str], eligible_nodes: List[dict]) -> Optional[dict]:
    """STAGE 2: if a prefix-hash match named a node that's still eligible,
    normally route back to it to keep its KV cache warm -- UNLESS thermal
    routing has priority over KV-cache warmth and that node is at/above
    the warning threshold, in which case the cache hit is deliberately
    bypassed to let it cool. Returns the matched node dict, or None if
    there was no match / it was bypassed (the caller then falls through
    to whichever routing strategy is configured)."""
    if not cached_node_name:
        return None

    thermal_cfg = state.CONFIG.get("thermal_routing", {})
    thermal_enabled = thermal_cfg.get("enabled", True)
    warning_temp = float(thermal_cfg.get("warning_temp_celsius", 70.0))
    thermal_over_kv = thermal_cfg.get("thermal_priority_over_kv_cache", True)

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
