"""Lightweight phase tracing for cross-rank scheduler/GPU timeline analysis."""

from __future__ import annotations

import contextlib
import os
from typing import Any

import torch

_ENABLED = os.environ.get("SGLANG_PHASE_TRACE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def is_phase_trace_enabled() -> bool:
    return _ENABLED


def phase_span(name: str, **fields: Any):
    if not _ENABLED:
        return contextlib.nullcontext()

    suffix = "".join(f" {key}={value}" for key, value in fields.items())
    return torch.profiler.record_function(f"phase::{name}{suffix}")
