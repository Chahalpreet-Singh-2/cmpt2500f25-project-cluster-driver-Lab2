import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.helpers import get_project_root, load_config, get_logger
from src.utils.model_utils import KProtoWrapper


log = get_logger()


class ModelPredictor:
    def __init__(self, model_path: Path):
        self.model = KProtoWrapper.load(model_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # expects same columns used during training
        return self.model.predict(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use trained K-Prototypes model to assign clusters."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to the training config YAML (default: configs/train_config.yaml)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help=(
            "Path to input data (parquet/csv). "
            "If not provided, uses train.in_parquet from the config."
        ),
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/kprototypes_model.joblib",
        help="Path to the saved K-Prototypes model (relative to project root).",
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default="data/processed/predictions.csv",
        help="Where to save the dataframe with Cluster labels.",
    )

    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    """Load a table from parquet or csv into a DataFrame."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")


def main() -> None:
    args = parse_args()
    root = get_project_root()

    # ---------- CONFIG ----------
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)

    # ---------- INPUT DATA ----------
    in_path_cfg = args.data or cfg["train"].get(
        "in_parquet",
        "data/processed/final_features.parquet",
    )
    in_path = Path(in_path_cfg)
    if not in_path.is_absolute():
        in_path = root / in_path
    in_path = in_path.resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input data not found: {in_path}")

    log.info(f"Loading data for prediction from: {in_path}")
    df = load_table(in_path)

    # ---------- FEATURE SELECTION ----------
    feature_cols = cfg["train"].get("feature_columns", [])
    keep = [c for c in feature_cols if c in df.columns]

    if not keep:
        raise ValueError(
            "No valid feature columns found in dataframe. "
            "Check 'feature_columns' in your config and columns in the input data."
        )

    work = df[keep].copy()

    # ---------- MODEL LOAD ----------
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = root / model_path
    model_path = model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    log.info(f"Loading model from: {model_path}")
    predictor = ModelPredictor(model_path)

    # ---------- PREDICT ----------
    log.info("Assigning clusters...")
    labels = predictor.predict(work)
    work["Cluster"] = labels

    # ---------- SAVE ----------
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv
    out_csv = out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    work.to_csv(out_csv, index=False)
    log.info(f"Saved predictions -> {out_csv}")


if __name__ == "__main__":
    main()

