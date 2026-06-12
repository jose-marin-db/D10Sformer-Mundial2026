"""D10Sformer training utilities.

Public API:
    Trainer              — step-based trainer with AMP + cosine LR
    TrainerConfig        — hyper-parameter dataclass
    LossSpec             — selects which heads contribute to the loss
    get_warmup_cosine_schedule — LR scheduler factory
    mlm_loss_from_logits, mlm_accuracy, perplexity_from_loss
"""

from .scheduler import get_warmup_cosine_schedule
from .training_metrics import mlm_loss_from_logits, mlm_accuracy, perplexity_from_loss
from .trainer import Trainer, TrainerConfig, LossSpec

__all__ = [
    "Trainer", "TrainerConfig", "LossSpec",
    "get_warmup_cosine_schedule",
    "mlm_loss_from_logits", "mlm_accuracy", "perplexity_from_loss",
]
