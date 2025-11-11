"""
Lightweight integration test for the clustering pipeline.

Goal: exercise the real CLI entrypoints (preprocess + train)
without running on the full 126k-row dataset.
"""

from pathlib import Path
import subprocess

import pandas as pd
import yaml
import pytest

from src.utils.helpers import get_project_root


@pytest.mark.integration
def test_end_to_end_cli_pipeline(tmp_path):
    """
    End-to-end: preprocess -> train (on a small sample) -> model file exists.

    This test:
      - Runs `python -m src.preprocess` if needed to create final_features.parquet
      - Creates a SMALL sampled parquet from the full features
      - Writes a temporary train_config.yaml that points to that small parquet
      - Runs `python -m src.train --config ... --save-model`
      - Asserts that the model file is created
    """
    root = get_project_root()

    # 1. Ensure preprocess has produced the big parquet at least once
    final_parquet = root / "data" / "processed" / "final_features.parquet"
    if not final_parquet.exists():
        subprocess.run(
            ["python", "-m", "src.preprocess"],
            check=True,
            cwd=root,
        )

    assert final_parquet.exists(), "final_features.parquet was not created by preprocess."

    # 2. Create a SMALL sample parquet for faster training
    df_full = pd.read_parquet(final_parquet)

    # Take up to 1000 rows for the integration test
    n_sample = min(1000, len(df_full))
    df_small = df_full.sample(n=n_sample, random_state=42)

    small_parquet = tmp_path / "small_features.parquet"
    df_small.to_parquet(small_parquet, index=False)

    # 3. Load the main train_config and override for test
    config_path = root / "configs" / "train_config.yaml"
    assert config_path.exists(), f"Config not found at {config_path}"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Override to use our small sample file and make training cheaper
    cfg.setdefault("train", {})
    cfg["train"]["in_parquet"] = str(small_parquet)
    cfg["train"]["n_clusters"] = 3  # fewer clusters -> faster
    cfg["train"]["save_model"] = True

    test_cfg_path = tmp_path / "train_config_test.yaml"
    with open(test_cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # 4. Run training CLI on the small sample
    subprocess.run(
        [
            "python",
            "-m",
            "src.train",
            "--config",
            str(test_cfg_path),
            "--save-model",
        ],
        check=True,
        cwd=root,
    )

    # 5. Check that a model file got created
    model_path = root / "models" / "kprototypes_model.joblib"
    assert model_path.exists(), f"Expected model file at {model_path}, but it does not exist."
