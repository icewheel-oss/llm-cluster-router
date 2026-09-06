# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import time
from typing import Dict, List

from router import state

# Rate limiting tracking: {client_identifier: [timestamp_1, timestamp_2, ...]}
RATE_LIMIT_CACHE: Dict[str, List[float]] = {}


def check_rate_limit(client_id: str) -> bool:
    """Checks if client exceeds configured requests per minute limit.
    Returns True if request is ALLOWED, False if RATE LIMITED.
    """
    rl_cfg = state.CONFIG.get("rate_limiting", {})
    if not rl_cfg.get("enabled", False):
        return True

    limit = rl_cfg.get("requests_per_minute", 60)
    now = time.time()

    history = RATE_LIMIT_CACHE.get(client_id, [])
    # Keep timestamps within the last 60 seconds
    history = [t for t in history if now - t < 60.0]

    if len(history) >= limit:
        return False

    history.append(now)
    RATE_LIMIT_CACHE[client_id] = history
    return True
