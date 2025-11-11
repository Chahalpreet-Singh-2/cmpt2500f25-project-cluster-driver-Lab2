"""
Tests for src/evaluate.py module.
Note: Uses 'y' for true labels and 'yhat' for predictions.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import src.evaluate as eval_mod
from src.evaluate import calculate_accuracy, calculate_metrics, evaluate_model


class TestCalculateAccuracy:
    """Tests for calculate_accuracy function."""

    def test_perfect_accuracy(self):
        """Test with perfect predictions."""
        y = np.array(["Yes", "No", "Yes", "No"])
        yhat = np.array(["Yes", "No", "Yes", "No"])

        accuracy = calculate_accuracy(y, yhat)

        assert accuracy == 1.0

    def test_half_accuracy(self):
        """Test with 50% accuracy."""
        y = np.array(["Yes", "No", "Yes", "No"])
        yhat = np.array(["Yes", "No", "No", "Yes"])

        accuracy = calculate_accuracy(y, yhat)

        assert accuracy == 0.5


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_all_metrics_present(self):
        """Test that all metrics are calculated."""
        y = np.array(["Yes", "No", "Yes", "No"] * 10)
        yhat = np.array(["Yes", "No", "Yes", "Yes"] * 10)

        metrics = calculate_metrics(y, yhat)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        # All metric values should be between 0 and 1
        assert all(0.0 <= v <= 1.0 for v in metrics.values())


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_complete_evaluation(self, trained_model, processed_data):
        """Test complete model evaluation."""
        X_test = processed_data["X_test"]
        y_test = processed_data["y_test"]

        results = evaluate_model(trained_model, X_test, y_test)

        # Basic checks
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1_score" in results

        # Optional richer outputs that your evaluate_model may return
        assert "confusion_matrix" in results
        assert "classification_report" in results
        # sanity checks on accuracy range
        assert 0.0 <= results["accuracy"] <= 1.0


def test_evaluate_main_smoke(tmp_path, monkeypatch):
    """
    Smoke-test src.evaluate.main() on a tiny numeric parquet file.
    It should run without crashing.
    """
    fake_root = tmp_path / "proj"
    (fake_root / "data" / "processed").mkdir(parents=True)

    # make sure we have at least as many rows as the max k used in KMeans
    n_rows = 20
    df = pd.DataFrame({
        "feature1": np.random.randn(n_rows),
        "feature2": np.random.randn(n_rows),
        "feature3": np.random.randn(n_rows),
    })
    parquet_path = fake_root / "data" / "processed" / "final_features.parquet"
    df.to_parquet(parquet_path, index=False)

    # monkeypatch project root in evaluate module
    monkeypatch.setattr(eval_mod, "get_project_root", lambda: fake_root)

    # ensure argparse sees just a simple argv
    monkeypatch.setattr(sys, "argv", ["evaluate"])

    # just verify it doesn't raise
    eval_mod.main()
