from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import os
import yaml

from mega_sena.db import load_history_df
from mega_sena.features import build_features
from mega_sena.hybrid import generate_predictions
from mega_sena.ml import prepare_model
from mega_sena.model_io import (
    load_model,
    save_model,
    simple_data_fingerprint,
)


def load_cfg(path: str = "config.yaml") -> dict:
    """Load config with robust defaults if missing/partial."""
    defaults = {
        "data_path": "data/mega_sena.csv",
        "db": {"path": "mega_sena.db"},
        "ml": {"n_estimators": 300, "max_depth": None, "random_state": 42},
        "hybrid": {"alpha": 0.5, "pool_size": 1000, "seed": 123, "top_k": 20},
        "api": {"host": "0.0.0.0", "port": 8000},
    }
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
            for k, v in user.items():
                if isinstance(v, dict) and k in defaults:
                    defaults[k].update(v)
                else:
                    defaults[k] = v
        except Exception:
            pass
    return defaults


class SuggestRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")  # ISO date
    contest: int
    k: int | None = None
    mode: Literal["hybrid", "rule", "ml"] = "hybrid"


class SuggestResponse(BaseModel):
    date: str
    contest: int
    mode: str
    alpha: float
    sequences: List[List[int]]
    rule_heads: List[int]
    ml_heads: List[int]


app = FastAPI(title="Mega-Sena Predictor (Heuristic)")

_STATE = {"df_feat": None, "model": None, "feat_cols": None, "cfg": load_cfg()}


@app.on_event("startup")
def _startup():
    cfg = _STATE["cfg"]
    db_path = cfg["db"]["path"]
    if not os.path.exists(db_path):
        raise RuntimeError(f"Database not found at '{db_path}'. Seed it first:  python main.py seed")

    # Load data and features
    df = load_history_df(db_path)
    df_feat = build_features(df)

    # Load cached model; retrain only if missing/stale; no holdout for faster boot
    model, manifest = load_model("models")
    need_retrain = model is None
    if manifest and not need_retrain:
        if simple_data_fingerprint(df) != manifest.get("data_hash"):
            need_retrain = True
        elif manifest.get("ml_config") != cfg["ml"]:
            need_retrain = True

    if need_retrain:
        model, feat_cols, _ = prepare_model(df_feat, cfg["ml"], use_holdout=False)
        save_model("models", model, feat_cols, cfg["ml"], simple_data_fingerprint(df))
    else:
        feat_cols = manifest["feature_columns"]

    _STATE.update({"df_feat": df_feat, "model": model, "feat_cols": feat_cols})


@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest):
    cfg = _STATE["cfg"]

    # Mode → alpha
    if req.mode == "rule":
        alpha = 1.0
    elif req.mode == "ml":
        alpha = 0.0
    else:
        alpha = float(cfg["hybrid"]["alpha"])

    k = req.k or cfg["hybrid"]["top_k"]

    try:
        top, expl = generate_predictions(
            _STATE["df_feat"],
            req.date,
            req.contest,
            _STATE["model"],
            _STATE["feat_cols"],
            top_k=k,
            alpha=alpha,
            pool_size=cfg["hybrid"]["pool_size"],
            seed=cfg["hybrid"]["seed"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return SuggestResponse(
        date=req.date,
        contest=req.contest,
        mode=req.mode,
        alpha=alpha,
        sequences=[list(x) for x in top],
        rule_heads=list(expl["weights"]["rule_head"]),
        ml_heads=list(expl["weights"]["ml_head"]),
    )


@app.post("/reload")
def reload_model():
    """Force retrain and refresh the cached model from current DB (no holdout)."""
    cfg = _STATE["cfg"]
    db_path = cfg["db"]["path"]
    if not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="DB missing. Run: python main.py seed")

    df = load_history_df(db_path)
    df_feat = build_features(df)
    model, feat_cols, _ = prepare_model(df_feat, cfg["ml"], use_holdout=False)
    save_model("models", model, feat_cols, cfg["ml"], simple_data_fingerprint(df))
    _STATE.update({"df_feat": df_feat, "model": model, "feat_cols": feat_cols})
    return {"status": "reloaded", "rows": len(df_feat)}
