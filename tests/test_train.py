"""
Tests for src/train.py module.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import src.train as train_mod
from src.train import load_data, with_overrides
from src.utils.model_utils import KProtoWrapper


# ---------- Tests for load_data ----------

class TestLoadData:
    """Tests for load_data function in src.train."""

    def test_load_data_csv(self, tmp_path):
        """CSV: should load as numpy array with correct shape."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        arr = load_data(str(csv_file))

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        # values as strings or numbers are fine; just check contents loosely
        assert arr[0, 0] in ("1", 1)

    def test_load_data_parquet(self, tmp_path):
        """Parquet: should load as numpy array."""
        parquet_file = tmp_path / "data.parquet"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_parquet(parquet_file, index=False)

        arr = load_data(parquet_file)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)

    def test_load_data_npy(self, tmp_path):
        """NPY: should load as numpy array."""
        npy_file = tmp_path / "data.npy"
        data = np.array([[1, 2], [3, 4]])
        np.save(npy_file, data)

        arr = load_data(npy_file)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, data)

    def test_load_data_unsupported_extension(self, tmp_path):
        """Unsupported extension should raise ValueError."""
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("dummy")

        with pytest.raises(ValueError, match="Unsupported data format"):
            load_data(txt_file)


# ---------- Tests for with_overrides (config handling) ----------

class TestWithOverrides:
    """Tests for with_overrides function in src.train."""

    def test_with_overrides_updates_train_section(self, tmp_path):
        """Override train.n_clusters and ensure config is updated."""
        cfg_file = tmp_path / "train_config.yaml"
        cfg_file.write_text(
            "train:\n"
            "  n_clusters: 5\n"
            "  init: Huang\n"
        )

        overrides = {"train": {"n_clusters": 10}}
        cfg = with_overrides(cfg_file, overrides)

        assert "train" in cfg
        assert cfg["train"]["n_clusters"] == 10
        assert cfg["train"]["init"] == "Huang"

    def test_with_overrides_adds_new_section(self, tmp_path):
        """Override section that didn't previously exist."""
        cfg_file = tmp_path / "train_config.yaml"
        cfg_file.write_text(
            "train:\n"
            "  n_clusters: 5\n"
        )

        overrides = {"evaluation": {"metric": "silhouette"}}
        cfg = with_overrides(cfg_file, overrides)

        assert "evaluation" in cfg
        assert cfg["evaluation"]["metric"] == "silhouette"


# ---------- Tests for KProtoWrapper (model_utils) ----------

class TestKProtoWrapper:
    """Tests for KProtoWrapper training and persistence."""

    def test_fit_on_small_mixed_dataframe(self):
        """KProtoWrapper should produce one label per row."""
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0],
                "num2": [10.0, 20.0, 30.0, 40.0],
                "make": ["A", "B", "A", "B"],  # categorical
            }
        )

        kp = KProtoWrapper(
            n_clusters=2,
            init="Huang",
            random_state=42,
            cat_cols=["make"],
        )

        labels = kp.fit(df)

        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(df)
        # cluster labels should be small integers 0..(k-1)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saved KProtoWrapper can be loaded and used again."""
        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0],
                "num2": [10.0, 20.0, 30.0, 40.0],
                "make": ["A", "B", "A", "B"],
            }
        )

        kp = KProtoWrapper(
            n_clusters=2,
            init="Huang",
            random_state=42,
            cat_cols=["make"],
        )
        labels_before = kp.fit(df)

        model_path = tmp_path / "kproto_model.joblib"
        kp.save(model_path)

        assert model_path.exists()

        kp_loaded = KProtoWrapper.load(model_path)
        # ensure loaded object has same cat_cols and can "predict" (refit)
        assert kp_loaded.cat_cols == ["make"]

        labels_after = kp_loaded.predict(df)
        assert len(labels_after) == len(df)
        # not strictly equal to labels_before, but should be valid clusters
        assert set(np.unique(labels_after)).issubset({0, 1})


# ---------- Optional: smoke test for using processed_data fixture ----------

def test_kproto_with_processed_data_numeric_only(processed_data):
    """
    Smoke test: convert processed_data X_train into a DataFrame with a fake
    categorical column and ensure KProtoWrapper runs.
    """
    X_train = processed_data["X_train"]
    df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
    # add a synthetic categorical column
    df["make"] = np.where(df.index % 2 == 0, "A", "B")

    kp = KProtoWrapper(
        n_clusters=3,
        init="Huang",
        random_state=42,
        cat_cols=["make"],
    )

    labels = kp.fit(df)
    assert len(labels) == len(df)


def test_train_main_writes_clustered_and_model(tmp_path, monkeypatch):
    # --- fake project root structure ---
    fake_root = tmp_path / "proj"
    (fake_root / "configs").mkdir(parents=True)
    (fake_root / "data" / "processed").mkdir(parents=True)

    # --- tiny feature parquet (20 rows, 3 cols) ---
    df = pd.DataFrame({
        "FSA_Code": ["A1A"] * 20,
        "make": ["Ford"] * 20,
        "price": np.linspace(10000, 20000, 20),
    })
    features_path = fake_root / "data" / "processed" / "final_features.parquet"
    df.to_parquet(features_path, index=False)

    # --- minimal train_config.yaml ---
    cfg_text = """train:
  in_parquet: data/processed/final_features.parquet
  out_clustered_csv: data/processed/clustered.csv
  save_model: true
  model_path: models/kprototypes_model.joblib
  feature_columns:
    - FSA_Code
    - make
    - price
  categorical_columns:
    - FSA_Code
    - make
  n_clusters: 2
  init: Huang
  random_state: 42
"""
    cfg_file = fake_root / "configs" / "train_config.yaml"
    cfg_file.write_text(cfg_text)

    # --- monkeypatch project root + path resolver in train module ---
    monkeypatch.setattr(train_mod, "get_project_root", lambda: fake_root)
    monkeypatch.setattr(train_mod, "resolve_under_root", lambda p: fake_root / p)

    # --- fix sys.argv so argparse doesn’t see pytest flags ---
    monkeypatch.setattr(sys, "argv", ["train"])  # no extra args

    # --- run main() ---
    train_mod.main()

    # --- assert outputs exist under fake root ---
    clustered = fake_root / "data" / "processed" / "clustered.csv"
    model_path = fake_root / "models" / "kprototypes_model.joblib"

    assert clustered.exists()
    assert model_path.exists()
