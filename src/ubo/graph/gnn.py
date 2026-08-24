"""A small graph convolution, for comparison against the engineered features.

Two layers of symmetric-normalised neighbourhood averaging followed by a linear
readout - the GCN propagation rule, written out in numpy so the comparison does
not depend on a framework and so every step is visible:

    H1 = relu(A_hat @ X @ W0)
    H2 = relu(A_hat @ H1 @ W1)
    y  = sigmoid(H2 @ w)

Trained with full-batch gradient descent on the labelled roots. The point of
including it is not to win; it is to have measured whether the topology carries
signal the hand-built features miss. On this graph it does not, and the
comparison in the report says so with the numbers that show it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .build import OwnershipGraph


def adjacency(graph: OwnershipGraph, node_ids: Sequence[str], symmetric: bool = True) -> np.ndarray:
    """Normalised adjacency with self-loops: D^-1/2 (A + I) D^-1/2."""
    index = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    a = np.eye(n, dtype=np.float64)
    for edge in graph.edges:
        i, j = index.get(edge.parent), index.get(edge.child)
        if i is None or j is None:
            continue
        weight = edge.confidence
        a[i, j] += weight
        if symmetric:
            a[j, i] += weight
    degree = a.sum(axis=1)
    degree[degree == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    return a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def node_features(graph: OwnershipGraph, node_ids: Sequence[str]) -> np.ndarray:
    """Local, topology-only descriptors.

    Deliberately *not* the engineered features from ``features.py``. Handing the
    GNN those would make the comparison meaningless: it would be testing the
    same hand-built signal through a different classifier rather than testing
    whether the topology carries anything on its own.
    """
    from ..config import SECRECY_JURISDICTIONS

    rows = []
    for node in node_ids:
        entity = graph.entities.get(node)
        out_edges = graph.children(node)
        in_edges = graph.parents(node)
        rows.append([
            float(len(out_edges)),
            float(len(in_edges)),
            float(sum(e.share_percent for e in out_edges) / 100.0),
            1.0 if graph.jurisdiction_of(node) in SECRECY_JURISDICTIONS else 0.0,
            1.0 if entity is not None and entity.entity_type == "person" else 0.0,
            float(len(entity.sources)) if entity is not None else 0.0,
            float(entity.weakest_link) if entity is not None else 1.0,
        ])
    matrix = np.asarray(rows, dtype=np.float64)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return (matrix - matrix.mean(axis=0)) / std


@dataclass
class GCNResult:
    scores: dict[str, float]
    train_loss: list[float]
    hidden_dim: int
    epochs: int
    n_parameters: int
    n_labels: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "parameters": self.n_parameters,
            "labelled_nodes": self.n_labels,
            "final_loss": round(self.train_loss[-1], 5) if self.train_loss else None,
            "parameters_per_label": round(self.n_parameters / self.n_labels, 1) if self.n_labels else None,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def train_gcn(
    graph: OwnershipGraph,
    labels: dict[str, bool],
    hidden_dim: int = 8,
    epochs: int = 300,
    lr: float = 0.05,
    weight_decay: float = 5e-3,
    seed: int = 3,
) -> GCNResult:
    node_ids = sorted(graph.entities)
    index = {n: i for i, n in enumerate(node_ids)}
    a_hat = adjacency(graph, node_ids)
    x = node_features(graph, node_ids)

    labelled = [(index[n], 1.0 if v else 0.0) for n, v in labels.items() if n in index]
    if not labelled:
        return GCNResult({}, [], hidden_dim, 0, 0, 0)
    idx = np.array([i for i, _ in labelled])
    y = np.array([v for _, v in labelled])

    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, 0.4, size=(x.shape[1], hidden_dim))
    w1 = rng.normal(0, 0.4, size=(hidden_dim, hidden_dim))
    w2 = rng.normal(0, 0.4, size=(hidden_dim, 1))

    losses: list[float] = []
    for _ in range(epochs):
        z1 = a_hat @ x @ w0
        h1 = np.maximum(z1, 0.0)
        z2 = a_hat @ h1 @ w1
        h2 = np.maximum(z2, 0.0)
        logits = (h2 @ w2).ravel()
        pred = _sigmoid(logits[idx])

        eps = 1e-9
        loss = float(-np.mean(y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps)))
        losses.append(loss)

        # Backward pass, restricted to the labelled rows.
        d_logits = np.zeros_like(logits)
        d_logits[idx] = (pred - y) / len(idx)
        d_w2 = h2.T @ d_logits[:, None] + weight_decay * w2
        d_h2 = d_logits[:, None] @ w2.T
        d_z2 = d_h2 * (z2 > 0)
        d_w1 = (a_hat @ h1).T @ d_z2 + weight_decay * w1
        d_h1 = a_hat.T @ d_z2 @ w1.T
        d_z1 = d_h1 * (z1 > 0)
        d_w0 = (a_hat @ x).T @ d_z1 + weight_decay * w0

        w2 -= lr * d_w2
        w1 -= lr * d_w1
        w0 -= lr * d_w0

    h1 = np.maximum(a_hat @ x @ w0, 0.0)
    h2 = np.maximum(a_hat @ h1 @ w1, 0.0)
    scores = _sigmoid((h2 @ w2).ravel())
    n_params = int(w0.size + w1.size + w2.size)
    return GCNResult(
        scores={n: float(scores[i]) for n, i in index.items()},
        train_loss=losses,
        hidden_dim=hidden_dim,
        epochs=epochs,
        n_parameters=n_params,
        n_labels=len(labelled),
    )


def evaluate_gcn(result: GCNResult, labels: dict[str, bool], threshold: float = 0.5) -> dict[str, Any]:
    scored = [(n, result.scores.get(n, 0.0), v) for n, v in labels.items() if n in result.scores]
    if not scored:
        return {"n": 0}
    tp = sum(1 for _n, s, v in scored if s >= threshold and v)
    fp = sum(1 for _n, s, v in scored if s >= threshold and not v)
    fn = sum(1 for _n, s, v in scored if s < threshold and v)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": len(scored),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        **result.to_dict(),
    }
