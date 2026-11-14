# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Temporary bridge that lets verl's VeOmni engine run on Intel XPU with an
unmodified VeOmni. See README.md in this folder for what each module bridges,
the upstream change it waits for, and how to delete the folder.

Set VERL_XPU_VEOMNI_BRIDGE=0 to skip the bridge (to test stock VeOmni).
"""

import logging
import os

logger = logging.getLogger(__name__)


def apply() -> None:
    """Apply every bridge. Idempotent; each module no-ops once VeOmni has the feature."""
    if os.getenv("VERL_XPU_VEOMNI_BRIDGE", "1") == "0":
        return
    from . import checkpoint_alias, device, float8, hub_fa2

    checkpoint_alias.apply()
    device.apply()
    hub_fa2.apply()
    float8.apply()  # opt-in via VERL_XPU_VEOMNI_FLOAT8
