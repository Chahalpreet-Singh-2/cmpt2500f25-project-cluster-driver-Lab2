"""
Tests for src.preprocess module.
"""

from pathlib import Path

import pandas as pd
import pytest

from src import preprocess
from src.preprocess import load_data, iqr_trim
from src import preprocess


class TestLoadData:
    """Tests for the load_data function."""

    def test_load_data_success_with_str_path(self, tmp_path):
        """load_data should read a valid CSV and return a non-empty DataFrame."""
        # Arrange: create a small CSV file
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\nval1,val2\nval3,val4\n")

        # Act
        df = load_data(str(csv_file))  # pass as string

        # Assert
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ["col1", "col2"]
        assert len(df) == 2
        assert df.iloc[0]["col1"] == "val1"
        assert df.iloc[0]["col2"] == "val2"

    def test_load_data_success_with_path_object(self, tmp_path):
        """load_data should also accept a Path object."""
        csv_file = tmp_path / "data2.csv"
        csv_file.write_text("a,b\n1,2\n3,4\n")

        df = load_data(csv_file)  # pass as Path

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2
        # we load dtype=str, so values are strings
        assert df.iloc[0]["a"] == "1"
        assert df.iloc[0]["b"] == "2"

    def test_load_data_file_not_found(self):
        """load_data should raise FileNotFoundError if the file does not exist."""
        fake_path = "this_file_definitely_does_not_exist.csv"

        with pytest.raises(FileNotFoundError):
            load_data(fake_path)


class TestIqrTrim:
    """Tests for the iqr_trim helper."""

    def test_iqr_trim_removes_extreme_outliers(self):
        """Rows with extreme values in selected columns should be dropped."""
        df = pd.DataFrame(
            {
                "price": [10, 11, 12, 13, 1000],  # 1000 is a big outlier
                "mileage": [100, 110, 120, 130, 140],
            }
        )

        trimmed = iqr_trim(df, ["price"])

        # Outlier should be gone
        assert 1000 not in trimmed["price"].values
        # We should have fewer rows than original
        assert len(trimmed) < len(df)

    def test_iqr_trim_ignores_missing_columns(self):
        """If a column isn't present, iqr_trim should just skip it."""
        df = pd.DataFrame({"price": [1, 2, 3]})

        trimmed = iqr_trim(df, ["not_a_column"])  # column doesn't exist

        # DataFrame should be unchanged
        pd.testing.assert_frame_equal(df, trimmed)



def test_main_combines_four_csvs_and_writes_parquet(tmp_path, monkeypatch):
    """
    Smoke test for preprocess.main():
    - create 4 tiny CBB_Listings_X.csv files under a fake project root
    - monkeypatch preprocess.get_project_root -> tmp_path
    - call preprocess.main()
    - assert that final_features.parquet is written with correct row count
    """
    fake_root = tmp_path
    raw_dir = fake_root / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Create 4 small CSVs that main() expects
    for i in range(1, 5):
        df = pd.DataFrame(
            {
                "make": ["Ford", "Toyota"],
                "price": [10000 + i, 12000 + i],
            }
        )
        (raw_dir / f"CBB_Listings_{i}.csv").write_text(df.to_csv(index=False))

    # 🔴 IMPORTANT: patch the symbol used inside src.preprocess
    monkeypatch.setattr(preprocess, "get_project_root", lambda: fake_root)

    # Act: run the main pipeline
    preprocess.main()

    # Assert: output parquet exists and has expected content
    out_parquet = fake_root / "data" / "processed" / "final_features.parquet"
    assert out_parquet.exists()

    df_out = pd.read_parquet(out_parquet)
    # 4 files × 2 rows each
    assert len(df_out) == 8
    assert set(df_out["make"].unique()) == {"Ford", "Toyota"}
