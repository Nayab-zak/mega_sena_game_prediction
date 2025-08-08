import numpy as np
from datetime import datetime
from .rules import generate_rule_pool
from .features import build_single_features
from .utils import score_combo_from_number_scores, combo_diversity_penalty, dedupe_keep_best, normalize
from .ml import predict_number_probs

def model_predict_safe(model, X_one_row):
    if hasattr(X_one_row, "shape") and X_one_row.shape[0] != 1:
        X_one_row = X_one_row.iloc[[0]]
    return predict_number_probs(model, X_one_row)

def generate_predictions(df_with_feats, date, contest, model, feature_columns,
                         top_k=20, alpha=0.5, pool_size=1000, seed=123):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    target_feats = build_single_features(date, contest)
    X_one = target_feats.reindex(columns=feature_columns, fill_value=0)

    pool, rule_w = generate_rule_pool(df_with_feats, target_feats, pool_size=pool_size, seed=seed)
    ml_probs = normalize(model_predict_safe(model, X_one))
    number_scores = normalize(alpha * rule_w + (1 - alpha) * ml_probs)

    scored = []
    for combo in pool:
        s = score_combo_from_number_scores(combo, number_scores, use_log=True)
        s -= combo_diversity_penalty(combo)
        scored.append((combo, s))

    scored = dedupe_keep_best(scored)
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, _ in scored[:top_k]]

    explanations = {
        "weights": {
            "rule_head": rule_w.argsort()[-10:][::-1] + 1,
            "ml_head": ml_probs.argsort()[-10:][::-1] + 1,
        },
        "alpha": alpha
    }
    return top, explanations
