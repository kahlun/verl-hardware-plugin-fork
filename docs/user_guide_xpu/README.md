# Intel XPU User Guide

Last updated: 06/04/2026.

## Introduction

This document describes how to use verl for reinforcement learning training on Intel XPU (Data Center GPU Max / Arc).

## Directory Structure

```
user_guide_xpu/
├── README.md              # This file
├── install_guidance.md    # Installation guide
├── quick_start.md         # Quick start
├── profiling.md           # Profiling guide
└── faq.md                 # FAQ and troubleshooting
```

## Getting Started

- [Installation Guide](./install_guidance.md)
- [Quick Start](./quick_start.md)
- [Profiling Guide](./profiling.md) — Use community torch profile on XPU
- [FAQ](./faq.md)

## Platform Summary

| Item | Description |
|------|-------------|
| Device type | `xpu` |
| Vendor identifier | `intel` |
| Communication backend | `xccl` (oneCCL) |
| Device visibility env var | `ZE_AFFINITY_MASK` |
| Ray resource name | `GPU` |
| IPC support | No |

## Environment Variables

```bash
export VERL_PLATFORM=intel
source /opt/intel/oneapi/setvars.sh
```
