"""
Pytest fixtures for the test suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_data():
    """
    Small sample dataframe for unit tests.
    """
    data = {
        "customerID": ["C001", "C002", "C003"],
        "gender": ["Male", "Female", "Male"],
        "tenure": [12, 24, 36],
        "MonthlyCharges": [50.5, 70.25, 25.0],
        "Churn": ["No", "No", "Yes"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def processed_data():
    """
    Fake preprocessed train/test data (numeric) for model tests.
    """
    np.random.seed(42)

    X_train = np.random.randn(80, 15)
    X_test = np.random.randn(20, 15)
    y_train = np.random.choice(["Yes", "No"], 80)
    y_test = np.random.choice(["Yes", "No"], 20)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Temporary output directory for tests that write files.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def trained_model(processed_data):
    """
    Simple RandomForest model trained on processed_data.
    Used by evaluation and prediction tests.
    """
    X_train = processed_data["X_train"]
    y_train = processed_data["y_train"]

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model
