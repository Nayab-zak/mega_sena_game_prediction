import numpy as np
from itertools import combinations

def digit_sum(n: int) -> int:
    return sum(int(d) for d in str(abs(int(n))))

def normalize(vec, eps=1e-9):
    v = np.asarray(vec, dtype=float)
    v = np.clip(v, eps, None)
    return v / v.sum()

def sample_combo_weighted(weights, rng):
    """Sample 6 unique numbers (1..60) without replacement, weighted by `weights`."""
    weights = normalize(weights)
    # Gumbel-top-k trick for weighted sampling without replacement
    g = -np.log(-np.log(rng.random(60)))
    keys = np.log(weights) + g
    picks = np.argsort(keys)[-6:][::-1]  # top-6
    return tuple(sorted((picks + 1).tolist()))

def combo_diversity_penalty(combo):
    """Encourage spread across decades and parity balance."""
    decades = [c // 10 for c in combo]
    decade_spread = len(set(decades))
    parity_balance = abs(sum(n % 2 for n in combo) - 3)  # 0 is best (3 even/3 odd)
    # Higher is worse → convert to small penalty subtracted from score
    return 0.03 * (3 - decade_spread) + 0.02 * parity_balance

def score_combo_from_number_scores(combo, number_scores, use_log=True):
    idx = np.array([n-1 for n in combo])
    vals = number_scores[idx]
    if use_log:
        return float(np.log(vals).sum())
    return float(vals.sum())

def dedupe_keep_best(candidates_with_scores):
    """Input: list[(combo_tuple, score)], keep best per combo."""
    best = {}
    for combo, s in candidates_with_scores:
        if combo not in best or s > best[combo]:
            best[combo] = s
    return [(c, s) for c, s in best.items()]
