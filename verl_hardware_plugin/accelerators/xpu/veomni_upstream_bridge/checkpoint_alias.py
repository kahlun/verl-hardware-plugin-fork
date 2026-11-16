# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""verl/VeOmni version skew (not XPU specific). Remove when verl updates its import.

VeOmni main moved ``veomni.models.checkpoint_tensor_loading`` to
``veomni.models.checkpoint.convert`` (VeOmni#1142, same functions); verl main still
imports the old path. Alias it so the unmodified core engine module imports.
"""

import importlib
import sys


def apply() -> None:
    try:
        importlib.import_module("veomni.models.checkpoint_tensor_loading")
    except ModuleNotFoundError:
        sys.modules["veomni.models.checkpoint_tensor_loading"] = importlib.import_module(
            "veomni.models.checkpoint.convert"
        )
