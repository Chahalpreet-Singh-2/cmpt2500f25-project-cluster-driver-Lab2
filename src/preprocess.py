from pathlib import Path
import pandas as pd
from src.utils.helpers import get_project_root, load_config, get_logger
from utils.data_utils import DataLoader, Cleaner
from utils.feature_utils import FeatureEngineer
import yaml
with open("configs/preprocess_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
raw_file = resolve_under_root(f"data/raw/{cfg['input_file']}")
log = get_logger()

def main():
    root = get_project_root()
    cfg = load_config(root / "configs" / "preprocess_config.yaml")

    dl = DataLoader(raw_path=root / "data" / "raw")
    df = dl.load("CBB_Listings.csv")
    coords = dl.load("POSTAL_CODE.csv")   # should contain latitude/longitude

    cleaner = Cleaner()
    df = cleaner.handle_basics(df)
    df = cleaner.iqr_trim(df, cfg.get("iqr_trim_columns", ['price','mileage','days_on_market']))

    fe = FeatureEngineer(coords_df=coords)
    df = fe.add_coordinates(df)
    df = fe.dealer_kpis(df)

    out_parquet = root / "data" / "processed" / "final_features.parquet"
    df.to_parquet(out_parquet, index=False)
    log.info(f"Saved {out_parquet}")

if __name__ == "__main__":
    main()
