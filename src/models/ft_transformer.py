"""FT-Transformer model structure for tabular feature tokenization and sequence modeling."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .transformer import TransformerEncoder


@dataclass
class FTD10SformerConfig:
    """Configuration class for FTD10Sformer."""
    vocab_size: int
    d_model: int = 128
    num_layers: int = 3
    num_heads: int = 4
    d_ff: int = 256
    num_tournament_classes: int = 200
    dropout: float = 0.15


class FeatureTokenizer(nn.Module):
    """Tokenizes heterogeneous tabular features into a sequence of continuous embeddings.
    
    Translates 3 categorical features (tournament, neutral, venue) and 9
    continuous features into 12 dense feature tokens.
    """

    def __init__(self, d_model: int = 128, num_tournament_classes: int = 200) -> None:
        super().__init__()
        # 3 Categorical feature embeddings
        self.tournament_emb = nn.Embedding(num_tournament_classes, d_model)
        self.neutral_emb = nn.Embedding(2, d_model)
        self.venue_emb = nn.Embedding(3, d_model)
        
        # 9 Continuous feature projections
        self.cont_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(9)
        ])
        
    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        e_tourn = self.tournament_emb(cat_features[:, 0])
        e_neut = self.neutral_emb(cat_features[:, 1])
        e_venue = self.venue_emb(cat_features[:, 2])
        e_cats = torch.stack([e_tourn, e_neut, e_venue], dim=1) # (B, 3, d_model)
        
        e_conts_list = []
        for i in range(9):
            val = cont_features[:, i].unsqueeze(-1)
            e_conts_list.append(self.cont_projections[i](val))
            
        e_conts = torch.stack(e_conts_list, dim=1) # (B, 9, d_model)
        return torch.cat([e_cats, e_conts], dim=1) # (B, 12, d_model)


class FTD10Sformer(nn.Module):
    """FT-Transformer with a 36-class score classification head.
    
    Transforms the sequence of categorical and continuous feature tokens
    using a standard Transformer encoder, then feeds the cls token representation
    to the joint 36-class ScoreHead.
    """

    def __init__(self, config: FTD10SformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = FeatureTokenizer(config.d_model, config.num_tournament_classes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        self.encoder = TransformerEncoder(
            num_layers=config.num_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        self.score_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 36)
        )
        
    def forward(self, cat_features: torch.Tensor, cont_features: torch.Tensor) -> torch.Tensor:
        B = cat_features.shape[0]
        feats = self.tokenizer(cat_features, cont_features)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, feats], dim=1) # (B, 13, d_model)
        
        out = self.encoder(x) # (B, 13, d_model)
        cls_out = out[:, 0, :] # (B, d_model)
        
        score_logits = self.score_head(cls_out)
        return score_logits
