# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

import random

from router.logging_setup import logger
from router.strategies import RoutingContext, register_strategy


@register_strategy("random")
def select_node(ctx: RoutingContext) -> dict:
    selected_node = random.choice(ctx.eligible_nodes)
    logger.info(f"Randomly selected node {selected_node['name']} for model '{ctx.requested_model}'")
    return selected_node
