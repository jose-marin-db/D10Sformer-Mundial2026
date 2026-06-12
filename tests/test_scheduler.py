"""Tests for src/training/scheduler.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.optim import SGD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.scheduler import get_warmup_cosine_schedule  # noqa: E402


def _toy_optimizer(lr: float = 1.0):
    param = torch.nn.Parameter(torch.zeros(1))
    return SGD([param], lr=lr)


def test_warmup_starts_near_zero():
    opt = _toy_optimizer(lr=1.0)
    sched = get_warmup_cosine_schedule(opt, num_warmup_steps=10, num_training_steps=100)
    # At step 0 the lr should be ~0 (or exactly 0 with this convention)
    sched.step()  # advances to step 1
    assert opt.param_groups[0]["lr"] < 0.2   # well below base lr


def test_warmup_reaches_base_lr_at_warmup_steps():
    opt = _toy_optimizer(lr=1.0)
    sched = get_warmup_cosine_schedule(opt, num_warmup_steps=10, num_training_steps=100)
    # Advance to step = warmup_steps
    for _ in range(10):
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, rel_tol=1e-6)


def test_cosine_decays_to_zero_by_end():
    opt = _toy_optimizer(lr=1.0)
    sched = get_warmup_cosine_schedule(opt, num_warmup_steps=0, num_training_steps=100)
    for _ in range(100):
        sched.step()
    # At/after the final step, lr should be ~0
    assert opt.param_groups[0]["lr"] < 1e-6


def test_min_lr_ratio_respected():
    opt = _toy_optimizer(lr=1.0)
    sched = get_warmup_cosine_schedule(opt, num_warmup_steps=0, num_training_steps=100, min_lr_ratio=0.1)
    for _ in range(100):
        sched.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, rel_tol=1e-3)


def test_invalid_args_raise():
    opt = _toy_optimizer()
    with pytest.raises(ValueError):
        get_warmup_cosine_schedule(opt, num_warmup_steps=-1, num_training_steps=10)
    with pytest.raises(ValueError):
        get_warmup_cosine_schedule(opt, num_warmup_steps=5, num_training_steps=0)
