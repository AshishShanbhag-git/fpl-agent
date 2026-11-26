"""Evaluation metrics for xPts predictions (pure Python, no numpy, so it is unit-testable anywhere).

Two kinds of metric, because the optimiser needs both:
* accuracy  - how close is each prediction?           MAE, RMSE, Pearson / Spearman correlation
* ranking   - does it put the right players on top?   mean actual points of the top-k predicted players
              per gameweek (k=1 is "the captain pick", k=10 is roughly "who you'd want in your XI").
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, Iterable, Sequence


def _check(pred: Sequence[float], actual: Sequence[float]) -> None:
    if len(pred) != len(actual):
        raise ValueError("pred and actual must have the same length")
    if not pred:
        raise ValueError("empty input")


def mae(pred: Sequence[float], actual: Sequence[float]) -> float:
    """Mean absolute error: on average, how many points off is a prediction?"""
    _check(pred, actual)
    return sum(abs(p - a) for p, a in zip(pred, actual)) / len(pred)


def rmse(pred: Sequence[float], actual: Sequence[float]) -> float:
    """Root mean squared error: like MAE but punishes big misses (a 15-point haul) more."""
    _check(pred, actual)
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(pred))


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Linear correlation in [-1, 1]. None if either side is constant (e.g. the 'predict 0' baseline)."""
    _check(x, y)
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks; ties get the average of the ranks they span (standard for Spearman)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation: does a higher prediction mean a higher actual score, regardless of scale?"""
    _check(x, y)
    return pearson(ranks(x), ranks(y))


def top_k_actual(groups: Iterable[Hashable], pred: Sequence[float], actual: Sequence[float],
                 k: int) -> float | None:
    """For each group (a gameweek), take the k rows with the highest prediction and average their actual
    points; then average over groups. Ties in the prediction are broken by row order, never by the actual
    score (that would be look-ahead)."""
    by_group: dict = defaultdict(list)
    for g, p, a in zip(groups, pred, actual):
        by_group[g].append((p, a))
    per_group = []
    for rows in by_group.values():
        top = sorted(enumerate(rows), key=lambda t: (-t[1][0], t[0]))[:k]
        per_group.append(sum(a for _, (_, a) in top) / len(top))
    return sum(per_group) / len(per_group) if per_group else None


def summarise(pred: Sequence[float], actual: Sequence[float]) -> dict:
    """All accuracy metrics in one dict (used per season x segment)."""
    return {
        "n": len(pred),
        "mae": mae(pred, actual),
        "rmse": rmse(pred, actual),
        "pearson": pearson(pred, actual),
        "spearman": spearman(pred, actual),
        "mean_pred": sum(pred) / len(pred),
        "mean_actual": sum(actual) / len(actual),
    }
