# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Plugin-side monkeypatches, one module per capability, applied only from
the platform class that needs them -- not from this package's `__init__.py`
and not from this plugin's shared top-level `verl_hardware_plugin/__init__.py`.

`PlatformXXX()` is only constructed once `verl.plugin.platform.
platform_manager._create_platform()` has actually selected that platform for
the process (see `PlatformXPU.__init__` in `platforms/platform_xpu.py`).
Applying a patch from a shared location instead would fire on any host where
the corresponding hardware/SDK merely happens to be importable, regardless of
which platform verl actually selects for the run.

Currently contains one module:

- `reduce_avg_allreduce_patch_xpu`: Intel XPU's `is_reduce_avg_supported`
  replacement. See its own docstring for what it does and why it's a
  monkeypatch rather than a `PlatformBase` hook (verl-hardware-plugin#26).
"""
