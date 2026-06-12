"""D10Sformer model components.

Public API:
    D10Sformer            — end-to-end model (encoder + multi-task heads)
    D10SformerConfig      — hyper-parameter dataclass
    MatchEmbedding        — token + position + segment embedding
    MultiHeadSelfAttention— attention layer (from scratch, Q/K/V/output)
    TransformerEncoder    — stack of Pre-LN encoder blocks
    TransformerEncoderBlock
    FeedForward
    MLMHead, ResultHead, ScoreHead — multi-task heads
"""

from .embeddings import MatchEmbedding, DEFAULT_NUM_SEGMENTS
from .attention import MultiHeadSelfAttention
from .transformer import FeedForward, TransformerEncoderBlock, TransformerEncoder
from .heads import MLMHead, ResultHead, ScoreHead, ClassificationHead
from .d10sformer import D10Sformer, D10SformerConfig
from .baselines import (
    BaselineModel,
    ELOBaseline,
    LogisticRegressionBaseline,
    XGBoostBaseline,
    LightGBMBaseline,
    get_all_baselines,
)

__all__ = [
    "D10Sformer", "D10SformerConfig",
    "MatchEmbedding", "DEFAULT_NUM_SEGMENTS",
    "MultiHeadSelfAttention",
    "FeedForward", "TransformerEncoderBlock", "TransformerEncoder",
    "MLMHead", "ResultHead", "ScoreHead", "ClassificationHead",
    "BaselineModel", "ELOBaseline", "LogisticRegressionBaseline",
    "XGBoostBaseline", "LightGBMBaseline", "get_all_baselines",
]
