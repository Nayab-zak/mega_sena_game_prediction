# mega_sena/ml.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import average_precision_score

BALLS = ["b1","b2","b3","b4","b5","b6"]

def make_xy(df_with_feats: pd.DataFrame):
    X = df_with_feats.drop(columns=["date"] + BALLS)
    y = pd.DataFrame(0, index=df_with_feats.index, columns=[f"n{i}" for i in range(1,61)], dtype=int)
    for idx, row in df_with_feats.iterrows():
        for c in BALLS:
            y.at[idx, f"n{int(row[c])}"] = 1
    return X, y

def _base_rf(n_estimators=300, max_depth=None, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

def train_model(df_with_feats: pd.DataFrame, n_estimators=300, max_depth=None, random_state=42):
    """Fit on ALL data (no holdout)."""
    X, y = make_xy(df_with_feats)
    model = MultiOutputClassifier(_base_rf(n_estimators, max_depth, random_state), n_jobs=-1)
    model.fit(X, y)
    return model, X.columns.tolist()

def train_with_holdout(df_with_feats: pd.DataFrame, val_fraction=0.2,
                       n_estimators=300, max_depth=None, random_state=42):
    """Chronological split: oldest -> train, most recent -> val."""
    df_sorted = df_with_feats.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    n_train = max(1, int((1 - val_fraction) * n))
    df_train = df_sorted.iloc[:n_train]
    df_val   = df_sorted.iloc[n_train:]

    X_tr, y_tr = make_xy(df_train)
    X_va, y_va = make_xy(df_val)

    model = MultiOutputClassifier(_base_rf(n_estimators, max_depth, random_state), n_jobs=-1)
    model.fit(X_tr, y_tr)

    # Probabilities on validation set
    y_hat = np.zeros_like(y_va.values, dtype=float)
    for j, est in enumerate(model.estimators_):
        proba = est.predict_proba(X_va)  # shape (m, 2)
        y_hat[:, j] = proba[:, 1]

    aps = []
    for j in range(y_va.shape[1]):
        try:
            ap = average_precision_score(y_va.iloc[:, j].values, y_hat[:, j])
        except ValueError:
            ap = np.nan
        aps.append(ap)

    metrics = {
        "val_fraction": val_fraction,
        "n_train": int(n_train),
        "n_val": int(n - n_train),
        "macro_avg_precision": float(np.nanmean(aps)),
        "per_label_ap": aps,
    }
    return model, X_tr.columns.tolist(), metrics

def predict_number_probs(model: MultiOutputClassifier, X_one):
    """Return np.array shape (60,) with probability each number (1..60) appears."""
    # ensure 2D input
    if hasattr(X_one, "shape") and X_one.shape[0] != 1:
        X_one = X_one.iloc[[0]]
    probs = []
    for est in model.estimators_:
        p = est.predict_proba(X_one)[0]  # [P(0), P(1)]
        probs.append(p[1])
    arr = np.asarray(probs, dtype=float)
    return np.clip(arr, 1e-6, 1.0)  # avoid log(0)

def prepare_model(df_with_feats: pd.DataFrame, cfg_ml: dict, use_holdout=True):
    """
    Single entry point used by CLI and API.
    Returns: (model, feature_columns, metrics_or_None)
    """
    if use_holdout:
        model, feat_cols, metrics = train_with_holdout(
            df_with_feats,
            val_fraction=0.2,
            n_estimators=cfg_ml.get("n_estimators", 300),
            max_depth=cfg_ml.get("max_depth", None),
            random_state=cfg_ml.get("random_state", 42),
        )
        return model, feat_cols, metrics
    else:
        model, feat_cols = train_model(
            df_with_feats,
            n_estimators=cfg_ml.get("n_estimators", 300),
            max_depth=cfg_ml.get("max_depth", None),
            random_state=cfg_ml.get("random_state", 42),
        )
        return model, feat_cols, None
