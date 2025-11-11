"""
Tests for src/predict.py module.
"""
import sys 
import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path
from src.predict import ModelPredictor
import src.predict as predict_module
import src.predict as predict_mod

class DummyModel:
    """
    Simple dummy model that mimics a fitted model
    with a .predict() method.
    """
    def __init__(self):
        self.called_with = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        self.called_with = df
        # return simple deterministic predictions
        return np.arange(len(df))


class TestModelPredictorInit:
    """Tests for ModelPredictor __init__."""

    def test_init_uses_get_project_root_and_kproto_load(
        self,
        monkeypatch,
        tmp_path,
    ):
        """
        Ensure ModelPredictor:
        - calls get_project_root()
        - constructs the correct absolute model path
        - calls KProtoWrapper.load with that path
        - stores the returned model
        """
        fake_root = tmp_path
        rel_model_path = "models/kprototypes_model.joblib"
        abs_model_path = fake_root / rel_model_path

        # make sure the directory exists (even though we won't really read it)
        abs_model_path.parent.mkdir(parents=True, exist_ok=True)
        abs_model_path.write_text("dummy contents")

        # 1) monkeypatch get_project_root to return our fake root
        monkeypatch.setattr(
            predict_module,
            "get_project_root",
            lambda: fake_root,
        )

        # 2) monkeypatch KProtoWrapper.load to check the path and return DummyModel
        dummy_model = DummyModel()

        def fake_load(path):
            # ensure it was called with exactly the resolved path
            assert str(path) == str(abs_model_path)
            return dummy_model

        monkeypatch.setattr(
            predict_module.KProtoWrapper,
            "load",
            staticmethod(fake_load),
        )

        # Act
        predictor = ModelPredictor(rel_model_path)

        # Assert
        assert predictor.model is dummy_model


class TestModelPredictorPredict:
    """Tests for ModelPredictor.predict()."""

    def test_predict_delegates_to_underlying_model(self, monkeypatch):
        """
        Ensure that ModelPredictor.predict:
        - calls the underlying model's predict()
        - returns a NumPy array of the right length
        - passes the same DataFrame through
        """
        dummy_model = DummyModel()

        # Monkeypatch __init__ so it doesn't try to load from disk;
        # instead, it just sets self.model to our dummy.
        def fake_init(self, model_path: str):
            self.model = dummy_model

        monkeypatch.setattr(
            predict_module.ModelPredictor,
            "__init__",
            fake_init,
        )

        predictor = ModelPredictor("ignored-path.joblib")
        df = pd.DataFrame({"a": [1, 2, 3]})

        preds = predictor.predict(df)

        # Assertions
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(df)
        # Check that dummy_model.predict got the exact same df
        assert dummy_model.called_with is df

    def test_predict_with_empty_dataframe(self, monkeypatch):
        """
        Optional behavior: decide what you want for empty input.
        Right now, ModelPredictor just forwards to model.predict,
        so we simulate a dummy that raises an error.
        """
        class ErrorModel(DummyModel):
            def predict(self, df):
                if df.empty:
                    raise ValueError("Input DataFrame is empty")
                return super().predict(df)

        def fake_init(self, model_path: str):
            self.model = ErrorModel()

        monkeypatch.setattr(
            predict_module.ModelPredictor,
            "__init__",
            fake_init,
        )

        predictor = ModelPredictor("ignored-path.joblib")
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            predictor.predict(empty_df)

class Dummy:
        def predict(self, X):
            return np.arange(len(X))
def test_load_model_roundtrip(tmp_path):
    model_path = tmp_path / "dummy.pkl"
    joblib.dump(Dummy(), model_path)

    loaded = predict_mod.load_model(str(model_path))
    assert hasattr(loaded, "predict")

def test_predict_helper_returns_array():
    X = np.random.randn(5, 3)
    yhat = predict_mod.predict(Dummy(), X)
    assert isinstance(yhat, np.ndarray)
    assert yhat.shape == (5,)

def test_predict_main_writes_predictions(tmp_path, monkeypatch):
    """Run predict.main() directly in-process to boost coverage."""
    # 1️ Create fake project layout
    fake_root = tmp_path / "proj"
    (fake_root / "data" / "processed").mkdir(parents=True)
    (fake_root / "models").mkdir(parents=True)

    # 2️ Small fake input parquet
    df = pd.DataFrame({
        "FSA_Code": ["A1A", "B2B", "C3C"],
        "make": ["Ford", "Toyota", "Honda"],
        "price": [10000.0, 12000.0, 9000.0],
    })
    features_path = fake_root / "data" / "processed" / "final_features.parquet"
    df.to_parquet(features_path, index=False)

    # 3️ Pretend project root is this fake one
    monkeypatch.setattr(predict_mod, "get_project_root", lambda: fake_root)

    # 4️ Fake predictor so we don’t need a real model
    class DummyPredictor:
        def __init__(self, model_path: str):
            self.model_path = model_path

        def predict(self, df):
            # return "clusters" (all zeros)
            return np.zeros(len(df), dtype=int)

    monkeypatch.setattr(predict_mod, "ModelPredictor", DummyPredictor)

    # 5️ Clean argv so argparse in your main() doesn’t see pytest flags
    monkeypatch.setattr(sys, "argv", ["predict"])

    # 6️ Run your main() directly
    predict_mod.main()

    # 7️ Verify an output file was written
    # (change this name if your script writes a different one)
    preds_path = fake_root / "data" / "processed" / "predictions.csv"
    assert preds_path.exists()

    # Optional: check contents
    out = pd.read_csv(preds_path)
    assert len(out) == len(df)

