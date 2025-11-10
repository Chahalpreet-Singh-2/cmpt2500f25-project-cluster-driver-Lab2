import mlflow
import mlflow.sklearn
from sklearn.metrics import (
 accuracy_score, precision_score, recall_score,
 f1_score, roc_auc_score, confusion_matrix
)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.model_utils import KProtoWrapper

# Try to import helper utilities; fall back to a minimal resolve_under_root if needed
try:
    from src.utils.helpers import (
        get_project_root,
        load_config,
        get_logger,
        resolve_under_root,
    )
except ImportError:
    from src.utils.helpers import get_project_root, load_config, get_logger  # type: ignore

    def resolve_under_root(path: str) -> Path:
        """
        Resolve a path relative to the project root (parent of src/), unless it's already absolute.
        """
        p = Path(path)
        if p.is_absolute():
            return p
        root = Path(__file__).resolve().parents[1]  # repo root = parent of src/
        return (root / p).resolve()


log = get_logger()


def load_data(path: str) -> np.ndarray:
    """
    Generic loader that returns a NumPy array from parquet/csv/npy.
    (Not currently used in main(), but kept here in case you need it later.)
    """
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).to_numpy()
    elif p.suffix == ".csv":
        return pd.read_csv(p).to_numpy()
    elif p.suffix == ".npy":
        return np.load(p)
    else:
        raise ValueError(f"Unsupported data format: {p.suffix}")


def with_overrides(path: str | Path, overrides: dict | None = None) -> dict:
    """
    Load a YAML config and optionally override specific keys.

    overrides should look like:
        {"train": {"n_clusters": 5}}
    """
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train K-Prototypes clustering model on processed features."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the training config YAML (default: configs/train_config.yaml)",
    )

    parser.add_argument(
        "-k",
        "--n-clusters",
        type=int,
        dest="k",
        help="Override the number of clusters (n_clusters) from config.",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        help="If set, save the trained model to the models/ directory.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = get_project_root()

    # Resolve config path under project root
    config_path = resolve_under_root(args.config)

    # If user passed -k/--n-clusters, override n_clusters in the config
    overrides = {"train": {"n_clusters": args.k}} if args.k else None
    cfg = with_overrides(config_path, overrides)

    # Get input parquet path from config (if present), otherwise use a default
    in_parquet = cfg["train"].get(
        "in_parquet",
        str(root / "data" / "processed" / "final_features.parquet"),
    )
    in_parquet = resolve_under_root(in_parquet)

    log.info(f"Loading processed features from: {in_parquet}")
    df = pd.read_parquet(in_parquet)

    # Keep only the feature columns that exist in df
    feature_cols = cfg["train"].get("feature_columns", [])
    keep = [c for c in feature_cols if c in df.columns]

    if not keep:
        raise ValueError(
            "No valid feature columns found in dataframe. "
            "Check 'feature_columns' in your config."
        )

    work = df[keep].copy()

    # Determine which columns are categorical
    cat_cols_cfg = cfg["train"].get("categorical_columns", [])
    cat_cols = [c for c in cat_cols_cfg if c in work.columns]

    log.info(f"Using {len(keep)} feature columns.")
    log.info(f"Categorical columns: {cat_cols}")

    # Create and fit K-Prototypes model
    n_clusters = cfg["train"]["n_clusters"]
    init = cfg["train"].get("init", "Cao")
    random_state = cfg["train"].get("random_state", 42)

    # Decide whether to save the model
    save_model_flag = args.save_model or cfg["train"].get("save_model", False)

    # 🔹 MLflow integration starts here
    mlflow.set_experiment("Clustering")

    with mlflow.start_run(run_name="kprototypes"):
        # Log parameters
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("init", init)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("feature_columns", ",".join(keep))
        mlflow.log_param("categorical_columns", ",".join(cat_cols))

        log.info(
            f"Training KProtoWrapper with n_clusters={n_clusters}, "
            f"init={init}, random_state={random_state}"
        )

        model = KProtoWrapper(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            cat_cols=cat_cols,
        )

        labels = model.fit(work)
        work["Cluster"] = labels

        # Try to log a simple metric (K-Prototypes cost)
        cost = getattr(model.model, "cost_", None)
        if cost is not None:
            mlflow.log_metric("kprototypes_cost", float(cost))

        # Save clustered data
        out_csv = root / "data" / "processed" / "clustered.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        work.to_csv(out_csv, index=False)
        log.info(f"Saved clustered data -> {out_csv}")

        # Log clustered data as artifact
        mlflow.log_artifact(str(out_csv), artifact_path="data")

        if save_model_flag:
            model_dir = root / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "kprototypes_model.joblib"
            model.save(model_path)
            log.info(f"Saved model -> {model_path}")

            # Log model file as artifact
            mlflow.log_artifact(str(model_path), artifact_path="models")
        else:
            log.info(
                "Model not saved (use --save-model flag or set train.save_model: true in config)."
            )
