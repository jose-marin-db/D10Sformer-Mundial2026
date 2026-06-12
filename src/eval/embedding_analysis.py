"""
Embedding analysis utilities for D10Sformer.

Extracts the static token embeddings from a trained model and provides
classical analyses (cosine similarity, vector analogies, k-NN, t-SNE/PCA
projection) that demonstrate whether the model learned semantically
meaningful representations of teams, players, and bucketed features.

References:
    Mikolov et al. (2013) "Distributed Representations of Words and Phrases"
        — vector analogies (King - Man + Woman ≈ Queen).
    van der Maaten & Hinton (2008) "Visualizing Data using t-SNE".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def get_token_embedding(model, vocab, token: str) -> torch.Tensor:
    """Returns the d_model-dim embedding vector of `token` from the model.

    Falls back to [UNK] if the token is not present. The returned tensor
    is detached (no grad) and on the same device as the model.
    """
    tid = vocab.encode(token)
    with torch.no_grad():
        emb = model.embeddings.token_embedding.weight[tid].detach().clone()
    return emb


def get_embeddings_matrix(model, vocab, tokens: Iterable[str]) -> tuple[list[str], torch.Tensor]:
    """Stack embeddings of multiple tokens into a (N, d_model) tensor.

    Returns:
        (resolved_tokens, embeddings) — `resolved_tokens` is the input list
        with unknown tokens dropped.
    """
    resolved = []
    vectors = []
    for t in tokens:
        if vocab.has(t):
            resolved.append(t)
            vectors.append(get_token_embedding(model, vocab, t))
    if not vectors:
        raise ValueError("None of the requested tokens are in the vocabulary.")
    return resolved, torch.stack(vectors, dim=0)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors (or matched batched tensors)."""
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())


def top_k_neighbours(
    model,
    vocab,
    query: str | torch.Tensor,
    k: int = 10,
    restrict_to: Optional[list[str]] = None,
    exclude_self: bool = True,
) -> list[tuple[str, float]]:
    """Returns the top-k tokens by cosine similarity to `query`.

    Args:
        query: token string OR a 1-D embedding vector.
        k: number of neighbours.
        restrict_to: optional list of candidate tokens. If None, search the
            full vocab.
        exclude_self: if True, drop the query itself from results (when query
            is a string).

    Returns:
        List of (token, cosine_similarity) sorted descending.
    """
    if isinstance(query, str):
        q_emb = get_token_embedding(model, vocab, query)
        self_token = query
    else:
        q_emb = query
        self_token = None

    if restrict_to is None:
        all_tokens = list(vocab.token_to_id.keys())
    else:
        all_tokens = [t for t in restrict_to if vocab.has(t)]

    resolved, mat = get_embeddings_matrix(model, vocab, all_tokens)
    # Normalised cosine
    q_norm = q_emb / (q_emb.norm() + 1e-12)
    mat_norm = mat / (mat.norm(dim=1, keepdim=True) + 1e-12)
    sims = (mat_norm @ q_norm).cpu().numpy()

    order = np.argsort(-sims)
    out = []
    for idx in order:
        tok = resolved[idx]
        if exclude_self and tok == self_token:
            continue
        out.append((tok, float(sims[idx])))
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Vector analogies — "A is to B as C is to ___"
# ---------------------------------------------------------------------------

def analogy_query(
    model, vocab,
    a: str, b: str, c: str,
    k: int = 5,
    restrict_to: Optional[list[str]] = None,
) -> list[tuple[str, float]]:
    """Mikolov-style analogy: query = vec(b) - vec(a) + vec(c).

    Returns the top-k tokens nearest to that query vector (excluding the
    inputs themselves).

    Example:
        analogy_query(model, vocab, 'TEAM_ARGENTINA', 'PLAYER_MESSI',
                      'TEAM_FRANCE') → ['PLAYER_MBAPPE', ...]
    """
    a_emb = get_token_embedding(model, vocab, a)
    b_emb = get_token_embedding(model, vocab, b)
    c_emb = get_token_embedding(model, vocab, c)
    query = b_emb - a_emb + c_emb

    raw = top_k_neighbours(model, vocab, query, k=k + 3, restrict_to=restrict_to,
                            exclude_self=False)
    # Drop inputs from results
    excluded = {a, b, c}
    filtered = [(t, s) for t, s in raw if t not in excluded]
    return filtered[:k]


# ---------------------------------------------------------------------------
# Dimensionality reduction (PCA + optional t-SNE)
# ---------------------------------------------------------------------------

@dataclass
class Reduction2D:
    """Result of a 2-D projection of embeddings."""
    tokens: list[str]
    coords: np.ndarray   # (N, 2)
    method: str          # 'pca' or 'tsne'
    explained_variance: Optional[tuple[float, float]] = None  # only for PCA


def pca_2d(model, vocab, tokens: Iterable[str]) -> Reduction2D:
    """Project embeddings to 2D via PCA. Returns Reduction2D."""
    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    X = mat.cpu().numpy()
    X = X - X.mean(axis=0, keepdims=True)   # centre
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T   # project on first 2 PCs

    total_var = (S ** 2).sum()
    ev = ((S[0] ** 2) / total_var, (S[1] ** 2) / total_var) if total_var > 0 else (0.0, 0.0)
    return Reduction2D(tokens=resolved, coords=coords, method="pca", explained_variance=ev)


def tsne_2d(
    model, vocab, tokens: Iterable[str],
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Reduction2D:
    """Project embeddings to 2D via t-SNE (uses sklearn).

    For visual clusters of teams/players. Less interpretable than PCA but
    typically much more readable in scatter plots.
    """
    from sklearn.manifold import TSNE
    resolved, mat = get_embeddings_matrix(model, vocab, tokens)
    X = mat.cpu().numpy()
    n = X.shape[0]
    if n < 4:
        raise ValueError(f"t-SNE needs >= 4 points; got {n}")
    perp = min(perplexity, max(5.0, n / 4 - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, init="pca")
    coords = tsne.fit_transform(X)
    return Reduction2D(tokens=resolved, coords=coords, method="tsne")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def relative_similarity_score(
    model, vocab,
    anchor: str,
    positive: str,
    negative: str,
) -> dict:
    """Returns cos(anchor, positive), cos(anchor, negative), and Δ.

    Δ > 0 means the model places `positive` closer to `anchor` than `negative`,
    which is the expected behaviour for semantically aligned pairs (e.g.,
    anchor=PLAYER_MESSI, positive=TEAM_ARGENTINA, negative=TEAM_FRANCE).
    """
    a = get_token_embedding(model, vocab, anchor)
    p = get_token_embedding(model, vocab, positive)
    n = get_token_embedding(model, vocab, negative)
    cs_p = cosine_sim(a, p)
    cs_n = cosine_sim(a, n)
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "cos_positive": cs_p,
        "cos_negative": cs_n,
        "delta": cs_p - cs_n,
        "correct": cs_p > cs_n,
    }
