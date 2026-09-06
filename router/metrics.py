# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Prometheus counters for the router's own routing decisions. See the
README's "Metrics (Prometheus)" section for what each one means.
`prometheus_client` is an optional dependency -- HAS_PROMETHEUS gates
every `.inc()` call site so a missing install never breaks routing, and
metrics_response() below returns an empty (still-200) body instead of
erroring.
"""
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

    ROUTER_KV_CACHE_BYPASS_COOLING = Counter(
        'router_kv_cache_bypass_cooling_total',
        'Total KV-cache hits bypassed for cooling at 70C warning threshold',
        ['node'],
    )
    ROUTER_THERMAL_REROUTES_COOLEST = Counter(
        'router_thermal_reroutes_coolest_total',
        'Total requests rerouted to coolest node when candidate nodes exceed warning temp',
        ['node'],
    )
    ROUTER_THERMAL_CRITICAL_HARD_BLOCKS = Counter(
        'router_thermal_critical_hard_blocks_total',
        'Total hard blocks triggered at 80C critical thermal ceiling',
        ['node'],
    )
    ROUTER_THERMAL_EVENTS = Counter(
        'router_thermal_events_total',
        'Total thermal routing intervention events',
        ['event_type'],
    )
    ROUTER_PREFIX_AFFINITY_ROUTED = Counter(
        'router_prefix_affinity_routed_total',
        'Total requests routed to a node based on prefix-hash affinity '
        '(does not guarantee the vLLM engine itself still had the KV blocks cached)',
        ['node'],
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


def metrics_response_body_and_type() -> tuple:
    """Returns (body, media_type) for the /metrics route. Empty body with
    a valid Prometheus media type if prometheus_client isn't installed --
    a scraper always gets 200, never an error, whether or not metrics are
    actually available."""
    if not HAS_PROMETHEUS:
        return "", "text/plain; version=0.0.4; charset=utf-8"
    return generate_latest(), CONTENT_TYPE_LATEST
