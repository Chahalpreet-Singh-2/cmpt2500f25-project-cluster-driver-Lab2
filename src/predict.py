import numpy as np
import pandas as pd
from utils.helpers import get_project_root
from utils.model_utils import KProtoWrapper

class ModelPredictor:
    def __init__(self, model_path: str):
        root = get_project_root()
        self.model = KProtoWrapper.load(root / model_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # expects same columns used during training
        return self.model.predict(df)

if __name__ == "__main__":
    # example
    mp = ModelPredictor("models/kprototypes_model.joblib")
    sample = pd.DataFrame([{}])  # TODO: put your feature columns here
    print(mp.predict(sample))
