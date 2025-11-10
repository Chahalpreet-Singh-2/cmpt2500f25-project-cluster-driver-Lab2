import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from src.utils.helpers import get_project_root, get_logger

log = get_logger()


def main():
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "final_features.parquet")

    # ---- numeric-only subset ----
    num = df.select_dtypes(include=np.number).fillna(0)

    # Subsample for speed
    max_samples = 5000
    if len(num) > max_samples:
        log.info(f"Subsampling from {len(num)} rows down to {max_samples} for evaluation.")
        num = num.sample(n=max_samples, random_state=42)

    X = StandardScaler().fit_transform(num)

    # Smaller k range to keep runtime reasonable
    ks = range(2, 11)
    sils = []

    for k in ks:
        log.info(f"Evaluating k={k}...")
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        sils.append(score)
        log.info(f"  silhouette={score:.4f}")

    best_idx = int(np.argmax(sils))
    best_k = ks[best_idx]
    best_score = sils[best_idx]

    log.info(f"Best silhouette (numeric proxy) ~ k={best_k}, score={best_score:.3f}")


if __name__ == "__main__":
    main()