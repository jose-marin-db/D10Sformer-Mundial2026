"""
Learning rate scheduler: linear warmup + cosine decay to zero.

Why this combination?

1. **Linear warmup** for the first `warmup_steps` updates. At step 0, the
   randomly-initialised model has high-variance gradients; an immediate full
   learning rate would destabilise. We ramp linearly from 0 to `base_lr`.

2. **Cosine decay** for the remaining steps. Smoothly decays the lr to 0
   following a half-cosine curve. This is the standard recipe in BERT
   (Devlin et al., 2018) and modern training (Loshchilov & Hutter, 2017
   "SGDR: Stochastic Gradient Descent with Warm Restarts"), and tends to
   produce a slightly smoother loss landscape near convergence than linear
   decay.

The function returns a `torch.optim.lr_scheduler.LambdaLR`, the standard
PyTorch way to define a custom step→lr function.
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """LinearWarmup → CosineDecay.

    Args:
        optimizer: the optimizer whose `lr` will be scaled.
        num_warmup_steps: linear warmup from 0 to base_lr over these steps.
        num_training_steps: total training steps (warmup + decay).
        min_lr_ratio: final lr = base_lr * min_lr_ratio (default 0 = decay to 0).
        last_epoch: for resuming training.

    Returns:
        A `LambdaLR` that PyTorch will call as `scheduler.step()` after each
        optimizer step.
    """
    if num_warmup_steps < 0 or num_training_steps <= 0:
        raise ValueError("num_warmup_steps must be >= 0 and num_training_steps > 0")

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            if num_warmup_steps == 0:
                return 1.0
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay over [num_warmup_steps, num_training_steps]
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
