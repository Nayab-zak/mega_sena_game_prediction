import os
import yaml
import click

from mega_sena.db import seed_from_csv, load_history_df
from mega_sena.features import build_features
from mega_sena.hybrid import generate_predictions
from mega_sena.ml import prepare_model
from mega_sena.model_io import (
    save_model,
    load_model,
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
            # fall back to defaults
            pass
    return defaults


@click.group()
def cli():
    """Mega-Sena heuristic predictor CLI."""
    pass


@cli.command(help="Create/refresh SQLite DB from CSV (A–H columns only).")
def seed():
    cfg = load_cfg()
    csv_path = cfg["data_path"]
    db_path = cfg["db"]["path"]
    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}")
    seed_from_csv(db_path, csv_path)
    click.echo(f"Database seeded → {db_path}")


@cli.command(help="Train and cache the model to ./models (with optional holdout metric).")
@click.option("--holdout/--no-holdout", default=True, show_default=True)
def train(holdout):
    cfg = load_cfg()
    db_path = cfg["db"]["path"]
    if not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}. Run: python main.py seed")

    df = load_history_df(db_path)
    df_feat = build_features(df)

    model, feat_cols, metrics = prepare_model(df_feat, cfg["ml"], use_holdout=holdout)
    if metrics:
        click.echo(
            f"[Holdout] train={metrics['n_train']}  val={metrics['n_val']}  "
            f"macro-AP={metrics['macro_avg_precision']:.4f}"
        )

    data_hash = simple_data_fingerprint(df)
    save_model("models", model, feat_cols, cfg["ml"], data_hash)
    click.echo("Model saved in ./models (model.joblib + manifest.json)")


@cli.command(help="Suggest sequences using cached model (retrain only if missing/stale).")
@click.option("--date", required=True, help="YYYY-MM-DD")
@click.option("--contest", required=True, type=int)
@click.option("--k", default=None, type=int, help="Override top_k from config")
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "rule", "ml"], case_sensitive=False),
    default="hybrid",
    show_default=True,
    help="Which approach to use."
)
@click.option("--force-retrain/--no-force-retrain", default=False, show_default=True)
@click.option("--holdout/--no-holdout", default=False, show_default=True, help="Only used if retraining occurs.")
def suggest(date, contest, k, mode, force_retrain, holdout):
    cfg = load_cfg()
    db_path = cfg["db"]["path"]
    if not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}. Run: python main.py seed")

    # Load & featurize history
    df = load_history_df(db_path)
    df_feat = build_features(df)

    # Try load model from cache
    model, manifest = load_model("models")
    need_retrain = force_retrain or (model is None)

    # Staleness checks (data or ML config changed)
    if manifest and not need_retrain:
        curr_hash = simple_data_fingerprint(df)
        if curr_hash != manifest.get("data_hash"):
            need_retrain = True
        elif manifest.get("ml_config") != cfg["ml"]:
            need_retrain = True

    if need_retrain:
        click.echo("Training (model missing/stale or --force-retrain)...")
        model, feat_cols, metrics = prepare_model(df_feat, cfg["ml"], use_holdout=holdout)
        if metrics:
            click.echo(
                f"[Holdout] train={metrics['n_train']}  val={metrics['n_val']}  "
                f"macro-AP={metrics['macro_avg_precision']:.4f}"
            )
        data_hash = simple_data_fingerprint(df)
        save_model("models", model, feat_cols, cfg["ml"], data_hash)
    else:
        feat_cols = manifest["feature_columns"]
        click.echo(f"Loaded cached model trained at {manifest.get('trained_at')}")

    # Mode → alpha
    alpha = 1.0 if mode.lower() == "rule" else 0.0 if mode.lower() == "ml" else float(cfg["hybrid"]["alpha"])
    k = int(k) if k is not None else int(cfg["hybrid"]["top_k"])

    top, expl = generate_predictions(
        df_feat, date, contest, model, feat_cols,
        top_k=k, alpha=alpha,
        pool_size=cfg["hybrid"]["pool_size"], seed=cfg["hybrid"]["seed"]
    )

    click.echo(f"\nMode: {mode.upper()} (alpha={alpha:.2f})  Date={date}  Contest={contest}")
    click.echo(f"Top {k} sequences:")
    for i, seq in enumerate(top, 1):
        click.echo(f"{i:02d}: {seq}")

    click.echo("\nHeads-up numbers:")
    click.echo(f"Rule-weighted: {list(expl['weights']['rule_head'])}")
    click.echo(f"ML-probable : {list(expl['weights']['ml_head'])}")


if __name__ == "__main__":
    cli()
