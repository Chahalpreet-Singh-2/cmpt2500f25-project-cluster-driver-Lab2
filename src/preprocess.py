from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# ---------- config ----------
CONFIG_PATH = Path("configs/preprocess_config.yaml")
with CONFIG_PATH.open("r") as f:
    cfg = yaml.safe_load(f)

# Expect YAML like:
# input_file: [CBB_Listings_1.csv, CBB_Listings_2.csv, CBB_Listings_3.csv, CBB_Listings_4.csv]
# iqr_trim_columns: [price, mileage, days_on_market]
# coords_file: POSTAL_CODE.csv
# out_parquet: data/processed/final_features.parquet

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

input_files = [RAW_DIR / f for f in cfg["input_file"]]
coords_file = RAW_DIR / cfg["coords_file"]
out_parquet = Path(cfg["out_parquet"])

# ---------- load & combine ----------
dfs = []
for f in input_files:
    if not f.exists():
        raise FileNotFoundError(f"Input file missing: {f}")
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)

# ---------- optional: join postal code coords ----------
if coords_file.exists():
    coords = pd.read_csv(coords_file)
    # assumes common key named 'POSTAL_CODE' (rename if your column differs)
    # e.g., df['POSTAL_CODE'] present in listings & coords
    key = "POSTAL_CODE"
    if key in df.columns and key in coords.columns:
        df = df.merge(coords, on=key, how="left")

# ---------- IQR trim helper ----------
def iqr_trim(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    trimmed = frame.copy()
    for c in cols:
        if c in trimmed.columns:
            q1 = trimmed[c].quantile(0.25)
            q3 = trimmed[c].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            trimmed = trimmed[(trimmed[c] >= low) & (trimmed[c] <= high)]
    return trimmed

# ---------- clean ----------
iqr_cols = cfg.get("iqr_trim_columns", [])
df = iqr_trim(df, iqr_cols)

# (Add any renames/type-casts/null handling your lab requires here)

# ---------- write ----------
out_parquet.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_parquet, index=False)
print(f"✅ Wrote {len(df):,} rows → {out_parquet}")
