import click
import os
import yaml

from mega_sena.db import seed_from_csv, load_history_df
from mega_sena.features import build_features
from mega_sena.hybrid import generate_predictions
from mega_sena.ml import prepare_model


def load_cfg(path="config.yaml"):
    """Load config with robust defaults if missing or partial."""
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


@cli.command(help="Suggest number sequences for a date/contest.")
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
@click.option(
    "--holdout/--no-holdout",
    default=True,
    show_default=True,
    help="Use chronological holdout to print a quick validation metric (slower)."
)
def suggest(date, contest, k, mode, holdout):
    cfg = load_cfg()
    db_path = cfg["db"]["path"]
    if not os.path.exists(db_path):
        raise SystemExit(
            f"Database not found: {db_path}\n"
            f"Run:  python main.py seed"
        )

    # Load & featurize history
    df = load_history_df(db_path)
    df_feat = build_features(df)

    # Determine alpha from mode (how much weight to rule vs ML)
    if mode.lower() == "rule":
        alpha = 1.0
        use_model = True   # still train model for heads-up numbers and consistency
    elif mode.lower() == "ml":
        alpha = 0.0
        use_model = True
    else:
        alpha = float(cfg["hybrid"]["alpha"])
        use_model = True

    # Train model (or skip if you truly want to, but we keep it for consistent outputs)
    model, feat_cols, metrics = prepare_model(df_feat, cfg["ml"], use_holdout=holdout)

    if metrics:
        click.echo(
            f"[Holdout] train={metrics['n_train']}  val={metrics['n_val']}  "
            f"macro-AP={metrics['macro_avg_precision']:.4f}"
        )

    # Generate predictions
    k = int(k) if k is not None else int(cfg["hybrid"]["top_k"])
    top, expl = generate_predictions(
        df_feat,
        date,
        contest,
        model,
        feat_cols,
        top_k=k,
        alpha=alpha,
        pool_size=cfg["hybrid"]["pool_size"],
        seed=cfg["hybrid"]["seed"],
    )

    click.echo(f"\nMode: {mode.upper()}  (alpha={alpha:.2f})")
    click.echo(f"Top {k} sequences for {date} (contest {contest}):")
    for i, seq in enumerate(top, 1):
        click.echo(f"{i:02d}: {seq}")

    click.echo("\nHeads-up numbers:")
    click.echo(f"Rule-weighted: {list(expl['weights']['rule_head'])}")
    click.echo(f"ML-probable : {list(expl['weights']['ml_head'])}")


if __name__ == "__main__":
    cli()
