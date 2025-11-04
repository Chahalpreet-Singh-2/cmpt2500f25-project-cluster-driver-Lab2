from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class DataLoader:
    raw_path: Path
    def load(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(self.raw_path / filename, on_bad_lines="skip")

class Cleaner:
    def handle_basics(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = ['dealer_email','dealer_phone','has_leather','has_navigation']
        df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
        df['listing_first_date'] = pd.to_datetime(df.get('listing_first_date'), errors='coerce')
        if 'days_on_market' in df:
            df['listing_dropoff_date'] = df['listing_first_date'] + pd.to_timedelta(df['days_on_market'], unit='D')
        if 'price' in df:
            med = df.loc[df['price'].ne(0), 'price'].median()
            df.loc[df['price'].eq(0), 'price'] = med
        for col in ['series','exterior_color','exterior_color_category','interior_color','interior_color_category']:
            if col in df and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        if 'certified' in df: df['certified'] = df['certified'].replace({0:'No',1:'Yes'})
        if 'wheelbase_from_vin' in df:
            df['wheelbase_from_vin'] = df['wheelbase_from_vin'].replace(0, np.nan)
            df['wheelbase_from_vin'] = df['wheelbase_from_vin'].fillna(df['wheelbase_from_vin'].median())
        return df

    def iqr_trim(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        for col in cols:
            if col in df.columns:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lb, ub = q1 - 1.5*iqr, q3 + 1.5*iqr
                df = df[(df[col] >= lb) & (df[col] <= ub)]
        return df
