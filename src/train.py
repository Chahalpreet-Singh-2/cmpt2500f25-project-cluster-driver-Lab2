from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.helpers import get_project_root, load_config, get_logger, resolve_under_root
from src.utils.model_utils import KProtoWrapper

cfg = with_overrides(args.config, {"train": {"n_clusters": args.k}} if args.k else None)
in_parquet = resolve_under_root(cfg["train"]["in_parquet"])
log = get_logger()

def load_data(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p).to_numpy()
    elif p.suffix == ".csv":
        return pd.read_csv(p).to_numpy()
    elif p.suffix == ".npy":
        return np.load(p)
    else:
        raise ValueError(f"Unsupported data format: {p.suffix}")

def main():
    root = get_project_root()
    cfg = load_config(root / "configs" / "train_config.yaml")
    df = pd.read_parquet(root / "data" / "processed" / "final_features.parquet")

    keep = [c for c in cfg["train"]["feature_columns"] if c in df.columns]
    work = df[keep].copy()
    cat_cols = [c for c in cfg["train"]["categorical_columns"] if c in work.columns]

    model = KProtoWrapper(n_clusters=cfg["train"]["n_clusters"],
                          init=cfg["train"]["init"],
                          random_state=cfg["train"]["random_state"],
                          cat_cols=cat_cols)
    labels = model.fit(work)
    work["Cluster"] = labels

    out_csv = root / "data" / "processed" / "clustered.csv"
    work.to_csv(out_csv, index=False)
    log.info(f"Saved clustered data -> {out_csv}")

    if cfg["train"].get("save_model", False):
        model_path = root / "models" / "kprototypes_model.joblib"
        model.save(model_path)
        log.info(f"Saved model -> {model_path}")

if __name__ == "__main__":
    main()
