"""Dataset and Collator for FT-Transformer model (D10Sformer v2)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FTDataset(Dataset):
    def __init__(self, df, tournament_map):
        self.df = df.reset_index(drop=True)
        self.tournament_map = tournament_map
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Categoricals: tourn_class, neutral, venue
        t_class = self.tournament_map.get(row['tournament_class'], 0)
        neutral = int(row['neutral'])
        
        venue_str = str(row['venue'])
        if venue_str == 'home':
            venue_val = 0
        elif venue_str == 'away':
            venue_val = 1
        else:
            venue_val = 2
        
        cat_feats = torch.tensor([t_class, neutral, venue_val], dtype=torch.long)
        
        # 9 Continuous features
        cont_cols = [
            'home_elo', 'away_elo', 'elo_diff',
            'home_form5_pts', 'away_form5_pts',
            'home_form5_gf', 'away_form5_gf',
            'home_form10_pts', 'away_form10_pts'
        ]
        # Fill missing values with default statistics
        cont_vals = []
        for col in cont_cols:
            val = row[col]
            if np.isnan(val):
                if 'elo' in col:
                    val = 1500.0
                elif 'gf' in col:
                    val = 1.0
                else:
                    val = 1.0
            cont_vals.append(float(val))
            
        cont_feats = torch.tensor(cont_vals, dtype=torch.float32)
        
        # Target score class index (0..35)
        h = int(min(max(row['score_h'], 0), 5))
        a = int(min(max(row['score_a'], 0), 5))
        score_idx = h * 6 + a
        
        return cat_feats, cont_feats, score_idx


class FTCollator:
    def __call__(self, batch):
        cat_list, cont_list, label_list = zip(*batch)
        cats = torch.stack(cat_list)
        conts = torch.stack(cont_list)
        labels = torch.tensor(label_list, dtype=torch.long)
        return cats, conts, labels
