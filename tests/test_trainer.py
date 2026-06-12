"""Tests for src/training/trainer.py.

These are smoke tests: we want to verify the trainer can do ≥10 steps on a
tiny synthetic dataset without NaNs or crashes, and that the loss actually
*decreases* on memorizable data (sanity check that gradients flow).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.vocabulary import FootballVocab  # noqa: E402
from data.tokenizer import MatchTokenizer, MatchDocument, PlayerRef  # noqa: E402
from data.dataset import MatchDataset  # noqa: E402
from data.collator import MLMCollator  # noqa: E402
from models.d10sformer import D10Sformer, D10SformerConfig  # noqa: E402
from training.trainer import Trainer, TrainerConfig, LossSpec  # noqa: E402


def _toy_vocab() -> FootballVocab:
    return FootballVocab.build_from_data(
        teams=["A", "B", "C", "D"],
        tournaments=["X", "Y"],
        stages=["group", "final"],
        player_appearances={f"100{i}": 50 for i in range(1, 30)},
        player_positions={f"100{i}": "FW" for i in range(1, 30)},
        k_player_threshold=10,
    )


def _toy_corpus() -> list[MatchDocument]:
    return [
        MatchDocument(tournament="X", team_a="A", team_b="B", venue="home",
                      result="home_win", home_score=2, away_score=1,
                      lineup_a=[PlayerRef(player_id=f"100{i+1}", position="FW") for i in range(11)]),
        MatchDocument(tournament="Y", team_a="C", team_b="D", venue="neutral",
                      result="draw", home_score=1, away_score=1,
                      lineup_a=[PlayerRef(player_id=f"100{i+1}", position="FW") for i in range(11)]),
        MatchDocument(tournament="X", team_a="B", team_b="C", venue="home",
                      result="away_win", home_score=0, away_score=2),
        MatchDocument(tournament="Y", team_a="D", team_b="A", venue="home",
                      result="home_win", home_score=3, away_score=0),
    ] * 8   # 32 docs


def _toy_model(vocab_size: int) -> D10Sformer:
    return D10Sformer(D10SformerConfig(
        vocab_size=vocab_size,
        d_model=32, num_layers=2, num_heads=2, d_ff=64,
        max_seq_length=128, num_segments=8,
        dropout=0.0, attention_dropout=0.0,
    ))


def _setup(tmp_dir: Path, max_steps: int = 20) -> Trainer:
    vocab = _toy_vocab()
    tokenizer = MatchTokenizer(vocab, max_seq_length=80)
    ds = MatchDataset(_toy_corpus(), tokenizer)
    collator = MLMCollator(vocab, mlm_probability=0.5, seed=42)   # high prob → strong signal
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collator)

    model = _toy_model(len(vocab))
    cfg = TrainerConfig(
        lr=1e-3, weight_decay=0.0, max_steps=max_steps,
        warmup_ratio=0.1, mixed_precision=False, log_every=5, eval_every=0,
        save_every=0, output_dir=str(tmp_dir), run_name="smoke",
        device="cpu",
    )
    return Trainer(model=model, train_loader=loader, config=cfg,
                   loss_spec=LossSpec(use_mlm=True))


def test_trainer_advances_steps():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _setup(Path(tmp), max_steps=10)
        trainer.train()
        assert trainer.step == 10


def test_trainer_writes_log():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _setup(Path(tmp), max_steps=10)
        trainer.train()
        assert trainer.log_path.exists()
        lines = trainer.log_path.read_text().strip().split("\n")
        assert len(lines) >= 1


def test_loss_decreases_on_memorizable_data():
    """With 4 unique docs repeated 8 times and high MLM prob, the model
    should be able to memorise and the loss should drop."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _setup(Path(tmp), max_steps=50)
        trainer.train()
        history = trainer.history
        assert len(history) >= 2
        first = history[0]["loss"]
        last = history[-1]["loss"]
        # Allow some noise but expect at least 5% decrease
        assert last < first * 0.95, f"Loss did not decrease: {first} → {last}"


def test_no_nan_in_loss():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _setup(Path(tmp), max_steps=20)
        trainer.train()
        for entry in trainer.history:
            assert entry["loss"] == entry["loss"], "NaN in loss"   # NaN != NaN trick


def test_checkpoint_save_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _setup(Path(tmp), max_steps=10)
        trainer.train()
        ckpt_path = trainer.output_dir / "final.pt"
        assert ckpt_path.exists()

        # Build a fresh trainer and reload — model state should match
        trainer2 = _setup(Path(tmp), max_steps=10)
        trainer2.load_checkpoint(ckpt_path)
        for (n1, p1), (n2, p2) in zip(trainer.model.named_parameters(),
                                        trainer2.model.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter mismatch at {n1}"
