import numpy as np
import pandas as pd
from collections import Counter
from .utils import normalize, sample_combo_weighted

NUMBERS = np.arange(1, 61)

SIM_FEATS = ["weekday","contest_mod_10","contest_mod_12","y_mod_60","dm_mod_60"]

def _similarity_weights(hist_feats: pd.DataFrame, target_feats: pd.DataFrame):
    tf = target_feats.iloc[0]
    # simple kernel: exact match gets 1, else 0.5 if close (for modular diffs)
    w = np.ones(len(hist_feats), dtype=float)
    w *= (hist_feats["weekday"] == tf["weekday"]).astype(float) * 0.5 + 0.5
    w *= (hist_feats["contest_mod_10"] == tf["contest_mod_10"]).astype(float) * 0.5 + 0.5
    w *= (hist_feats["contest_mod_12"] == tf["contest_mod_12"]).astype(float) * 0.5 + 0.5
    w *= (hist_feats["y_mod_60"] == tf["y_mod_60"]).astype(float) * 0.3 + 0.7
    w *= (hist_feats["dm_mod_60"] == tf["dm_mod_60"]).astype(float) * 0.3 + 0.7
    return w

def rule_number_weights(df_with_feats: pd.DataFrame, target_feats: pd.DataFrame, l2=1.0):
    hist_feats = df_with_feats[target_feats.columns.tolist()]
    w = _similarity_weights(hist_feats, target_feats)
    counts = np.zeros(60, dtype=float)
    for i, row in df_with_feats.iterrows():
        weight = w[i]
        for col in ["b1","b2","b3","b4","b5","b6"]:
            n = int(row[col])
            counts[n-1] += weight
    # regularize toward uniform
    counts = counts + l2
    return normalize(counts)

def generate_rule_pool(df_with_feats: pd.DataFrame, target_feats: pd.DataFrame, pool_size=800, seed=0):
    weights = rule_number_weights(df_with_feats, target_feats)
    rng = np.random.default_rng(seed)
    pool = set()
    while len(pool) < pool_size:
        combo = sample_combo_weighted(weights, rng)
        pool.add(combo)
    return list(pool), weights
