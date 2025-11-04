from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

@dataclass
class KProtoWrapper:
    n_clusters: int = 10
    init: str = "Huang"
    random_state: int = 42
    scaler: StandardScaler = StandardScaler()
    model: object = None
    cat_cols: list = None

    def fit(self, df: pd.DataFrame):
        from kmodes.kprototypes import KPrototypes
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        df_scaled = df.copy()
        if num_cols:
            df_scaled[num_cols] = self.scaler.fit_transform(df_scaled[num_cols])
        cols_order = self.cat_cols + [c for c in df_scaled.columns if c not in self.cat_cols]
        arr = df_scaled[cols_order].values
        categorical_indices = list(range(len(self.cat_cols)))
        self.model = KPrototypes(n_clusters=self.n_clusters, init=self.init, n_init=5,
                                 verbose=1, random_state=self.random_state)
        labels = self.model.fit_predict(arr, categorical=categorical_indices)
        return labels

    def predict(self, df: pd.DataFrame):
        # K-Prototypes doesn't expose a pure predict on new mixed data reliably;
        # usually re-fit or assign by nearest mode. For demo we re-fit on combined.
        return self.fit(df)

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler, "cat_cols": self.cat_cols}, path)

    @staticmethod
    def load(path):
        bundle = joblib.load(path)
        kp = KProtoWrapper()
        kp.model = bundle["model"]; kp.scaler = bundle["scaler"]; kp.cat_cols = bundle["cat_cols"]
        return kp
