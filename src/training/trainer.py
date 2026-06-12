"""
Trainer for D10Sformer.

Design philosophy:
- **Step-based**, not epoch-based. Modern Transformer training tracks steps
  (= optimizer updates) because epochs are not commensurable across runs
  (different batch sizes → different #steps per epoch). All schedules,
  evaluations and checkpoints are triggered by step count.
- **Composable losses**. The trainer accepts a `LossSpec` that chooses which
  of {mlm, result, score} losses to combine, with their weights. Pre-training
  uses only MLM; fine-tuning uses MLM (auxiliary) + result + score.
- **Mixed precision** via `torch.amp.autocast` + `GradScaler`. On T4 this
  doubles throughput at no accuracy cost for our model size.
- **Local JSON logging**. No external services — every N steps we append a
  metrics dict to a JSONL file. Vic can plot these from any notebook.
- **Resumable checkpoints**. Save model + optimizer + scheduler + step.

References:
    Devlin et al. (2018) BERT
    Loshchilov & Hutter (2017, 2019) SGDR / Decoupled WD
    Micikevicius et al. (2018) Mixed Precision Training
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .scheduler import get_warmup_cosine_schedule
from .training_metrics import mlm_loss_from_logits, mlm_accuracy, perplexity_from_loss


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------

@dataclass
class LossSpec:
    """Which heads contribute to the loss and with what weight.

    Optional class weights for the Result and Score heads — useful when the
    target distribution is imbalanced (e.g., 'draw' is the minority class
    in football data). When provided, they are passed to `F.cross_entropy`
    as the `weight` argument, multiplying per-sample loss by the weight of
    the true class. The standard recipe is sklearn-style inverse-frequency:

        w_i = n_total / (k * n_i)

    where k = num_classes and n_i = count of class i in the training set.
    """
    use_mlm: bool = True
    use_result: bool = False
    use_score: bool = False
    lambda_mlm: float = 1.0
    lambda_result: float = 1.0
    lambda_score: float = 0.3
    # Optional class weights (lists; converted to tensors inside the Trainer)
    result_class_weights: list[float] | None = None
    score_class_weights: list[float] | None = None


@dataclass
class TrainerConfig:
    # Optimisation
    lr: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Training duration
    max_steps: int = 1000
    accumulation_steps: int = 1
    mixed_precision: bool = True

    # Logging / eval / checkpoint
    log_every: int = 20
    eval_every: int = 200
    save_every: int = 500
    save_best: bool = True

    # Paths
    output_dir: str = "./checkpoints"
    run_name: str = "smoke_test"

    # Reproducibility
    seed: int = 42

    # Device override (auto if None)
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Step-based trainer for D10Sformer.

    Args:
        model: a D10Sformer (or any nn.Module returning
            {result_logits, score_logits, mlm_logits} from its forward).
        train_loader: DataLoader yielding CollatedBatch.
        val_loader: optional DataLoader for evaluation.
        config: TrainerConfig.
        loss_spec: which losses to combine.
        on_step_end: optional callback(step, metrics_dict) for custom hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: TrainerConfig = TrainerConfig(),
        loss_spec: LossSpec = LossSpec(),
        on_step_end: Optional[Callable[[int, dict], None]] = None,
    ):
        self.config = config
        self.loss_spec = loss_spec
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.on_step_end = on_step_end

        # Device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Optimizer (decoupled weight decay; biases and LN params get wd=0)
        self.optimizer = self._build_optimizer()

        # Scheduler
        warmup_steps = int(config.warmup_ratio * config.max_steps)
        self.scheduler = get_warmup_cosine_schedule(
            self.optimizer, warmup_steps, config.max_steps
        )

        # AMP scaler (no-op when mixed_precision=False or device=cpu)
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Cache class-weight tensors on the device (if provided in loss_spec).
        # Computing them in __init__ avoids re-allocating on every forward.
        self._result_weights_tensor = None
        if loss_spec.result_class_weights is not None:
            self._result_weights_tensor = torch.tensor(
                loss_spec.result_class_weights, device=self.device, dtype=torch.float32
            )
        self._score_weights_tensor = None
        if loss_spec.score_class_weights is not None:
            self._score_weights_tensor = torch.tensor(
                loss_spec.score_class_weights, device=self.device, dtype=torch.float32
            )

        # State
        self.step = 0
        self.best_val_loss = float("inf")
        self.history: list[dict] = []

        # Output dir
        self.output_dir = Path(config.output_dir) / config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metrics.jsonl"
        # Truncate previous log
        self.log_path.write_text("")

    # ----- optimizer construction -----

    def _build_optimizer(self) -> AdamW:
        """AdamW with weight decay only on Linear/Embedding weights.

        Biases, LayerNorm gains/biases and embedding scale params get wd=0.
        This is the BERT/HuggingFace convention.
        """
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return AdamW(
            groups, lr=self.config.lr,
            betas=self.config.betas, eps=self.config.eps,
        )

    # ----- main loop -----

    def train(self) -> None:
        """Run training for `config.max_steps` steps."""
        self.model.train()
        accumulation_counter = 0
        running_loss = 0.0
        running_mlm_acc = 0.0
        running_n = 0
        t_start = time.time()

        infinite_loader = self._infinite(self.train_loader)
        while self.step < self.config.max_steps:
            batch = next(infinite_loader)
            batch = self._batch_to_device(batch)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, parts = self._compute_loss(batch)

            loss = loss / self.config.accumulation_steps
            self.scaler.scale(loss).backward()

            accumulation_counter += 1
            running_loss += float(loss.item()) * self.config.accumulation_steps
            running_n += 1
            running_mlm_acc += parts.get("mlm_acc", 0.0)

            if accumulation_counter == self.config.accumulation_steps:
                # Unscale + clip + step
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.step += 1
                accumulation_counter = 0

                if self.step % self.config.log_every == 0:
                    avg_loss = running_loss / running_n
                    avg_acc = running_mlm_acc / running_n
                    elapsed = time.time() - t_start
                    metrics = {
                        "step": self.step,
                        "lr": self.scheduler.get_last_lr()[0],
                        "loss": avg_loss,
                        "mlm_perplexity": perplexity_from_loss(avg_loss),
                        "mlm_acc": avg_acc,
                        "elapsed_s": round(elapsed, 1),
                        "steps_per_s": round(self.step / elapsed, 2),
                    }
                    self._log(metrics)
                    running_loss = 0.0
                    running_mlm_acc = 0.0
                    running_n = 0

                # Eval
                if (
                    self.val_loader is not None
                    and self.config.eval_every > 0
                    and self.step % self.config.eval_every == 0
                ):
                    val_metrics = self.evaluate()
                    val_metrics["step"] = self.step
                    self._log({"phase": "eval", **val_metrics})
                    if self.config.save_best and val_metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_loss"]
                        self.save_checkpoint(self.output_dir / "best.pt")

                # Periodic checkpoint
                if self.config.save_every > 0 and self.step % self.config.save_every == 0:
                    self.save_checkpoint(self.output_dir / f"step_{self.step}.pt")

                if self.on_step_end is not None:
                    self.on_step_end(self.step, {"loss": avg_loss if running_n == 0 else None})

        # Final checkpoint
        self.save_checkpoint(self.output_dir / "final.pt")

    # ----- forward / loss -----

    def _compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        out = self.model(
            token_ids=batch.token_ids,
            segment_ids=batch.segment_ids,
            attention_mask=batch.attention_mask,
        )
        parts: dict[str, float] = {}
        total = torch.tensor(0.0, device=self.device)

        if self.loss_spec.use_mlm:
            mlm_loss = mlm_loss_from_logits(out["mlm_logits"], batch.mlm_labels)
            total = total + self.loss_spec.lambda_mlm * mlm_loss
            acc, n = mlm_accuracy(out["mlm_logits"].detach(), batch.mlm_labels)
            parts["mlm_loss"] = float(mlm_loss.item())
            parts["mlm_acc"] = acc

        if self.loss_spec.use_result:
            res_loss = F.cross_entropy(
                out["result_logits"], batch.result_labels,
                weight=self._result_weights_tensor,
                ignore_index=-100,
            )
            total = total + self.loss_spec.lambda_result * res_loss
            parts["result_loss"] = float(res_loss.item())

        if self.loss_spec.use_score:
            sco_loss = F.cross_entropy(
                out["score_logits"], batch.score_labels,
                weight=self._score_weights_tensor,
                ignore_index=-100,
            )
            total = total + self.loss_spec.lambda_score * sco_loss
            parts["score_loss"] = float(sco_loss.item())

        return total, parts

    # ----- evaluation -----

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> dict:
        """Compute val_loss, val_mlm_acc, val_perplexity over the full loader."""
        loader = loader or self.val_loader
        if loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_n = 0
        n_batches = 0
        for batch in loader:
            batch = self._batch_to_device(batch)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                loss, parts = self._compute_loss(batch)
            total_loss += float(loss.item())
            n_batches += 1
            if self.loss_spec.use_mlm:
                acc, n = mlm_accuracy(
                    self.model(batch.token_ids, batch.segment_ids,
                               attention_mask=batch.attention_mask)["mlm_logits"],
                    batch.mlm_labels,
                )
                total_acc += acc * n
                total_n += n

        self.model.train()
        avg_loss = total_loss / max(1, n_batches)
        avg_acc = (total_acc / total_n) if total_n > 0 else 0.0
        return {
            "val_loss": avg_loss,
            "val_mlm_perplexity": perplexity_from_loss(avg_loss),
            "val_mlm_acc": avg_acc,
        }

    # ----- utilities -----

    def _batch_to_device(self, batch):
        # CollatedBatch has .to()
        return batch.to(self.device)

    def _infinite(self, loader: DataLoader) -> Iterable:
        """Generator that wraps loader and restarts when exhausted."""
        while True:
            for batch in loader:
                yield batch

    def _log(self, metrics: dict) -> None:
        self.history.append(metrics)
        with self.log_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        # Compact console line
        if "phase" in metrics and metrics["phase"] == "eval":
            print(
                f"[eval @ step {metrics.get('step'):>6}] "
                f"val_loss={metrics.get('val_loss', 0):.4f}  "
                f"val_ppl={metrics.get('val_mlm_perplexity', 0):.2f}  "
                f"val_acc={metrics.get('val_mlm_acc', 0):.4f}"
            )
        else:
            print(
                f"[step {metrics['step']:>6}/{self.config.max_steps}] "
                f"lr={metrics['lr']:.2e}  "
                f"loss={metrics['loss']:.4f}  "
                f"ppl={metrics['mlm_perplexity']:.2f}  "
                f"mlm_acc={metrics['mlm_acc']:.4f}  "
                f"({metrics['steps_per_s']:.1f} step/s)"
            )

    def save_checkpoint(self, path: Path) -> None:
        ckpt = {
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "loss_spec": asdict(self.loss_spec),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.step = ckpt["step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
