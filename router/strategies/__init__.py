# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Pluggable routing-strategy registry.

A strategy decides which node handles a request, once the pool of
`eligible_nodes` has already been narrowed down (model-served-here check,
thermal Stage 1 critical-block filter, and prefix-affinity Stage 2 have
all already run -- a strategy only ever sees nodes that survived those).

To add a built-in strategy: create a module in this package implementing
`select_node(ctx: RoutingContext) -> dict`, decorate it with
`@register_strategy("your-name")`, and import that module at the bottom
of this file so it self-registers.

To add a strategy WITHOUT touching this project's source (e.g. a
self-hosted deployment's own custom logic): put it in any importable
module implementing the same shape, and list that module's dotted path
in the LLM_ROUTER_EXTRA_STRATEGY_MODULES env var (comma-separated) --
router/app.py imports each one at startup, and each module's own
`@register_strategy(...)` call does the rest. `routing.mode` in
config.yaml then just needs to match whatever name you registered.

Note what this does and doesn't make hot-reloadable: which ALREADY-
REGISTERED strategy is active, and any strategy's own tunable config
(read fresh from CONFIG on every request, e.g. `routing.sticky_header`),
change with a plain config.yaml edit -- no restart, via the existing
hot-reload mechanism (router/state.py). Registering brand new strategy
CODE for the first time still requires one process start, same as any
Python import -- no framework safely hot-loads arbitrary new code into a
running process, and this doesn't pretend otherwise.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class RoutingContext:
    eligible_nodes: List[dict]
    requested_model: str
    prompt: str
    json_data: dict
    headers: Dict[str, str]
    auth_user: str
    client_ip: str
    routing_config: Dict[str, Any] = field(default_factory=dict)


class RoutingStrategy(Protocol):
    def __call__(self, ctx: RoutingContext) -> dict: ...


_REGISTRY: Dict[str, RoutingStrategy] = {}


def register_strategy(name: str):
    """Decorator: registers a `select_node(ctx) -> dict` function under
    `name`, matching a `routing.mode` value in config.yaml."""
    def _wrap(fn: RoutingStrategy) -> RoutingStrategy:
        _REGISTRY[name] = fn
        return fn
    return _wrap


def get_strategy(name: str) -> RoutingStrategy:
    """Looks up a strategy by its `routing.mode` name. Raises ValueError
    (listing what IS registered) rather than silently falling back to a
    default -- an unrecognized mode should fail loudly at request time,
    not quietly misroute every request."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown routing.mode '{name}'. Registered strategies: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_strategy_names() -> List[str]:
    """All currently-registered strategy names, sorted. Surfaced in
    `POST /_router/reload`'s response so an operator can confirm a
    newly-added extra strategy module actually loaded."""
    return sorted(_REGISTRY)


def find_header_case_insensitive(headers: Dict[str, str], name: str) -> str:
    """Header dicts here are plain dicts (from FastAPI's `dict(request.headers)`),
    not case-insensitive multidicts -- this does the same manual lookup the
    sticky/smart strategies both need, once."""
    value = headers.get(name)
    if value:
        return value
    for h_name, h_val in headers.items():
        if h_name.lower() == name:
            return h_val
    return ""


# Built-in strategies self-register by being imported here.
from router.strategies import random_strategy, smart, sticky  # noqa: E402,F401
