# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Regression test: MLU's profiler patch must not break other platforms.

verl-core's own ``Profiler.start()`` (``verl/utils/profiler/torch_profile.py``) calls
``get_torch_profiler(..., profile_step=..., schedule=...)`` on every profiled step, on
every platform. Before the delegate fix, MLU's hand-copied reimplementation of
``get_torch_profiler`` didn't know about ``profile_step``/``name_mini_batch_window`` and
TypeError'd on this call for every platform, not just MLU, the moment
``verl_hardware_plugin`` was installed -- confirmed live against real verl 0.9.1.
"""

import verl.utils.profiler.torch_profile as tp
from verl_hardware_plugin.profilers import torch_profile_mlu


def test_get_torch_profiler_accepts_new_kwargs_for_non_mlu_contents(tmp_path):
    torch_profile_mlu._patch_get_torch_profiler()

    prof = tp.get_torch_profiler(
        contents=["cpu"],
        save_path=str(tmp_path),
        role="actor",
        rank=0,
        profile_step=3,
        name_mini_batch_window=True,
    )

    assert prof is not None
