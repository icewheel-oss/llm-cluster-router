# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""The simplest possible strategy -- no state, no config, uniform choice
among whatever nodes survived model-availability and thermal filtering.
The fallback `routing.mode` if nothing more specific is configured."""
import random

from router.logging_setup import logger
from router.strategies import RoutingContext, register_strategy


@register_strategy("random")
def select_node(ctx: RoutingContext) -> dict:
    """Picks uniformly at random from ctx.eligible_nodes."""
    selected_node = random.choice(ctx.eligible_nodes)
    logger.info(f"Randomly selected node {selected_node['name']} for model '{ctx.requested_model}'")
    return selected_node
