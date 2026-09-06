# Copyright (c) 2026 Rohit Khatkar
# Licensed under the MIT License (see LICENSE for details)

"""Single place logging is configured, imported by every other module in
this package so there's exactly one `logging.basicConfig()` call. See
audit_log.py for why AUDIT_LOG entries use a separate, dedicated logger.
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("llm-cluster-router")

# Dedicated logger for AUDIT_LOG entries (per-request completion records).
# Same handler/format as `logger` today, so stdout output is byte-identical
# to before -- this exists so a consumer can attach their own handler (a
# separate file, a different verbosity, routed to a different sink) to
# just "llm-cluster-router.audit" via standard logging config, instead of
# grep'ing the combined operational+audit stream for the "AUDIT_LOG: "
# string prefix.
audit_logger = logging.getLogger("llm-cluster-router.audit")
