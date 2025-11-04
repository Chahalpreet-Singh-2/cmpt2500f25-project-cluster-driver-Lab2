from dataclasses import dataclass
import pandas as pd

@dataclass
class FeatureEngineer:
    coords_df: pd.DataFrame  # must contain dealer_postal_code, latitude, longitude

    def add_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        df['dealer_postal_code'] = df['dealer_postal_code'].astype(str)
        coords = self.coords_df.rename(columns={'dealer_postal_code ':'dealer_postal_code'})
        keep = ['dealer_postal_code','latitude','longitude']
        coords = coords[[c for c in keep if c in coords.columns]].drop_duplicates()
        return df.merge(coords, on='dealer_postal_code', how='left')

    def dealer_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        sold = df[df['listing_type'].eq('Sold')].copy()
        if sold.empty: return df
        tvs = sold.groupby('dealer_name')['listing_type'].count().rename('total_vehicles_sold')
        brand = (sold.groupby(['dealer_name','make'])['listing_type']
                    .count().rename('brand_vehicles_sold').reset_index())
        idx = brand.groupby('dealer_name')['brand_vehicles_sold'].idxmax()
        msb = brand.loc[idx][['dealer_name','make']].rename(columns={'make':'most_sold_brand'})
        sold['revenue'] = sold['price']
        rev = sold.groupby('dealer_name')['revenue'].sum().rename('total_revenue')
        dom = df.groupby('dealer_name')['days_on_market'].mean().rename('avg_days_on_market')
        return (df.drop(columns=['total_vehicles_sold','most_sold_brand','total_revenue','avg_days_on_market'],
                        errors='ignore')
                  .merge(tvs, left_on='dealer_name', right_index=True, how='left')
                  .merge(msb, on='dealer_name', how='left')
                  .merge(rev, left_on='dealer_name', right_index=True, how='left')
                  .merge(dom, left_on='dealer_name', right_index=True, how='left'))
