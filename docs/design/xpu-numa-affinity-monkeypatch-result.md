# XPU NUMA affinity: does the pure monkeypatch work?

**Short answer: yes. It works. Verified on real B60 hardware on 2026-09-24,
with zero changes to verl-core.**

The confusing part is that the first hardware run *failed*. That failure had
nothing to do with monkeypatching — it was a too-old `pyzes` version in the
container image. Fixed that, re-ran, everything passed. This doc separates
those two things, because they are easy to mix up.

---

## 1. The question this answers

Two open PRs were doing one fix (making NUMA pinning work on Intel XPU):

| PR | Layer | What it does |
|---|---|---|
| [kahlun/verl#19](https://github.com/kahlun/verl/pull/19) | verl **core** | Adds a `PlatformBase.set_numa_affinity()` hook and makes `verl/utils/distributed.py` call it |
| [plugin-fork#9](https://github.com/kahlun/verl-hardware-plugin-fork/pull/9) | **plugin** | Implements that hook for XPU using pyzes |

Plugin PR #9 does nothing without core PR #19 — the hook it overrides doesn't
exist upstream, so nothing would ever call it.

The question was: **can this be done plugin-only, with a monkeypatch, so the
core PR isn't needed at all?** Per the maintainer's ruling on
[plugin issue #26](https://github.com/kahlun/verl-hardware-plugin-fork/issues/26)
(2026-09-23), plugin-side monkeypatch is the default and a core PR is only for
genuine hardware-agnostic blockers.

## 2. The answer

Yes. The branch `experiment/xpu-numa-affinity-monkeypatch` does it with **no
verl-core diff at all**, and it is verified on hardware, not just argued from
source.

```
[PASS] verl.utils.distributed.set_numa_affinity is patched
[PASS] verl selected the XPU platform            device_name='xpu' vendor='intel'
[PASS] engine_workers' from-imported binding is the patched function
[PASS] resolved PCI BDF via pyzes                bdf=0000:3d:00.0 numa_node=0
[PASS] affinity pinning changed this process's cpuset    512 cpus -> 256 cpus
6 passed, 0 failed, 0 skipped
```

Three of those lines matter more than the others:

- **"is patched"** — the plugin loaded through its normal `verl.plugins`
  entry point and replaced the core function. Nothing in verl-core was
  touched.
- **"engine_workers' from-imported binding is the patched function"** — this
  is the claim the whole approach rests on (see §5a), and it is the one that
  a silent failure would have hidden. Now confirmed live.
- **"512 cpus -> 256 cpus"** — the pinning actually happened. The process was
  restricted to the 256 logical CPUs local to its GPU's NUMA node. Before
  this patch, that call was a silent no-op on XPU.

## 3. Why the first run failed (and why it isn't a monkeypatch problem)

First run on B60:

```
[PASS] verl.utils.distributed.set_numa_affinity is patched
[PASS] engine_workers' from-imported binding is the patched function
[FAIL] resolved PCI BDF via pyzes
       AttributeError: module 'pyzes' has no attribute 'zes_pci_properties_t'
```

Read the pass/fail split carefully: **the monkeypatch machinery all passed on
the very first attempt.** The patch installed, the platform check worked, the
import-order claim held. What failed was the line that talks to pyzes — i.e.
the pyzes code copied verbatim from PR #9.

Cause: the image ships **pyzes 0.1.1**, which contains *no PCI symbols at
all*. `zesDevicePciGetProperties`, `zes_pci_properties_t` and
`zes_pci_address_t` were added by
[oneapi-src/level-zero#462](https://github.com/oneapi-src/level-zero/pull/462)
and first released in **0.1.2** (2026-06-12). Confirmed by diffing
`dir(pyzes)` between the two versions in the same container:

```
pyzes 0.1.1 -> pci-related attributes: []            # none, at all
pyzes 0.1.2 -> zesDevicePciGetProperties, zes_pci_properties_t,
               zes_pci_address_t, zesDevicePciGetState, ...
```

A dependency-version problem, not a design problem. Installing
`pyzes>=0.1.2` and re-running gave the clean 6/6 above.

### This is also a real bug in PR #9

Worth calling out because it applies to the `PlatformBase`-hook version
identically: that `AttributeError` gets swallowed into a warning, so on an
image with pyzes 0.1.1 the run just stays **silently un-pinned** — exactly
the failure mode of the original bug being fixed. `_zes_device_pci_bdf()` now
checks for the symbol up front and raises a message naming the version it
needs.

## 4. What this means for the PRs

**Recommendation: close core PR #19 and plugin PR #9, ship the plugin-only
branch instead.**

1. It's now demonstrated that no core change is needed — so #19 doesn't clear
   the "genuine hardware-agnostic blocker" bar from the issue #26 ruling.
2. Core PR #19 buys CUDA **nothing**. It's a pure refactor: `PlatformBase`'s
   default implementation is today's pynvml code verbatim. The one real bug it
   fixed (the `libnuma.so` probe left in the dispatcher) was *introduced by
   that refactor*, not pre-existing. Hard to pitch upstream as "fixes
   something."
3. One repo to review instead of two coordinated PRs.

What's given up: if upstream ever renames `set_numa_affinity` or changes its
signature, the plugin patch silently stops matching — no merge conflict to
warn you. That surface is small (one function), unlike the ~40-line
constructor forks that made "monkeypatch everything" unattractive for the
rollout-server profiler case, and the patch logs loudly when it can't find
what it expects.

## 5. The two non-obvious things that make it work

Both of these would otherwise have caused a silent no-op.

### (a) Both call sites import the function *by value*

```python
# verl/workers/engine_workers.py:39
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity

# verl/model_merger/megatron_model_merger.py:44
from verl.utils.distributed import set_numa_affinity
```

`from X import y` binds `y` into the importing module's own namespace at
import time. So reassigning `verl.utils.distributed.set_numa_affinity` later
does **not** reach those call sites. The patch has to land *before* those two
modules are imported.

It does, and not by luck: `verl/__init__.py` loads plugin entry_points after
importing only `.protocol`, `.utils.device`, `.utils.import_utils` and
`.utils.logging_utils` — none of which reach `engine_workers` or
`model_merger`. And importing any `verl.*` submodule always runs
`verl/__init__.py` first, so the ordering is guaranteed. There is also a
`_rebind_importers()` safety net that fixes up those two modules if they ever
do get imported first.

This is the thing the hardware run proved rather than assumed.

### (b) The patch must NOT be applied from `PlatformXPU.__init__`

Every *other* patch in `verl_hardware_plugin/patches/` is applied from
`PlatformXPU.__init__`, deliberately: only patch once verl has actually
selected XPU, so a mixed host with `VERL_PLATFORM=nvidia` isn't affected.

That reasoning breaks for this one, because of two adjacent lines in
`verl/model_merger/megatron_model_merger.py`:

```python
set_numa_affinity()  # line 154
torch.distributed.init_process_group(get_nccl_backend())  # line 155
```

`get_nccl_backend()` is what would first call `get_platform()` and therefore
first construct `PlatformXPU` — *one line too late*.
`BaseModelMerger.__init__` never touches the platform either. So on the
checkpoint-merge path, a patch applied from `PlatformXPU.__init__` would never
have been installed at all.

So it is applied from `patches.apply_all()` at plugin-import time, and the
"is XPU selected?" check moves from *apply* time to *call* time. That keeps
CUDA and every other platform on their exact existing pynvml path. It can't be
done the other way round either: calling `get_platform()` during plugin import
would freeze the platform singleton before anything could call
`set_platform()`.

## 6. What's in the branch

Worktree: `/home/sdp/lun/fork/verl-hardware-plugin-numa-monkeypatch-wt`
Branch: `experiment/xpu-numa-affinity-monkeypatch`, based on PR #8
(`experiment/xpu-monkeypatch-no-core-hooks`) — **not pushed yet**.

| File | What |
|---|---|
| `verl_hardware_plugin/patches/numa_affinity_patch_xpu.py` | The patch. pyzes code reused from PR #9, plus the version guard |
| `verl_hardware_plugin/patches/__init__.py` | Registers it in `apply_all()` |
| `tests/test_numa_affinity_patch_xpu.py` | 24 tests, passing. verl and pyzes stubbed, so it runs on a CPU-only host |
| `scripts/verify_numa_patch_xpu.py` | The on-hardware check that produced §2 |
| `scripts/devctl_verify_numa.sh` | devctl driver: overlays this branch onto the image, ensures pyzes>=0.1.2, runs the check |
| `docs/design/xpu-monkeypatch-experiment.md` | Updated with the NUMA row and the full verification section |

### How to reproduce the hardware run

devctl cannot sync a git worktree (in a worktree `.git` is a *file*, and the
tar-based sync walks it as a directory and dies). So stage a `.git`-free
mirror first. Also note `devctl session sync` and `devctl test` default to the
`workspace:` key in `~/.devctl/config.yaml`, **not** the session's own
workspace — pass `--workspace=` explicitly, then `--skip-sync` on the run, or
the PVC gets overwritten with the wrong tree.

```bash
STAGE=/home/sdp/lun/fork/numa-monkeypatch-devctl-stage
rsync -a --delete --exclude='.git' --exclude='__pycache__' \
  /home/sdp/lun/fork/verl-hardware-plugin-numa-monkeypatch-wt/ "$STAGE"/
devctl session sync --workspace="$STAGE"
devctl test --skip-sync \
  --image=amr-registry.caas.intel.com/devctl/verl-intel-gpu:pr8-monkeypatch-20260924 \
  --gpu=2 --gpu-model=B60 --cpu=16 --memory=64Gi --shm-size=8Gi \
  --command="bash,/workspace/local/scripts/devctl_verify_numa.sh" --attach
```

## 7. Still to do

- [ ] **`pyzes>=0.1.2` needs adding to
      `docker/intel_gpu/requirements-intel-gpu.txt`** on the
      `feature/xpu-docker` branch
      ([PR #2](https://github.com/kahlun/verl-hardware-plugin-fork/pull/2)).
      That file currently pins neither pyzes nor pynvml, so the image picks up
      0.1.1 transitively and NUMA pinning silently does nothing.
- [ ] Push the branch / open a PR, and close #19 and #9 if §4 is agreed.
- [ ] Optional: run inside a real GRPO training job rather than the standalone
      check, to see the pin happen per worker under Ray. Watch item there:
      `_resolve_local_rank()` returns the *global* device id Ray assigned,
      while pyzes enumerates only process-visible devices. They agree when
      `RAY_EXPERIMENTAL_NOSET_ZE_AFFINITY_MASK` is set and can disagree when
      Ray sets `ZE_AFFINITY_MASK` per worker. The pynvml path being replaced
      has the identical property, so it is not a regression here — but the
      verification run had `ZE_AFFINITY_MASK` unset, so it did not exercise
      the disagreeing case.
- [ ] Note for whoever owns it: the image `pr8-monkeypatch-20260924` was built
      from an **unpushed** local worktree (`fork/vhp-pr8-dockertest-wt`)
      containing a `_xpu_guard.py` refactor that is not on any branch. If that
      lands, note this patch deliberately does *not* use its apply-time
      `xpu_available()` guard — see §5b for why it checks at call time instead.
