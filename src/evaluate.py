import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from utils.helpers import get_project_root, get_logger

log = get_logger()

def main():
    root = get_project_root()
    df = pd.read_parquet(root / "data" / "processed" / "final_features.parquet")
    num = df.select_dtypes(include=np.number).fillna(0)
    X = StandardScaler().fit_transform(num)

    ks = range(2, 15)
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init='auto', random_state=42).fit(X)
        sils.append(silhouette_score(X, km.labels_))
    best_k = ks[int(np.argmax(sils))]
    log.info(f"Best silhouette (numeric proxy) ~ k={best_k}, score={max(sils):.3f}")

if __name__ == "__main__":
    main()
