import os
import numpy as np
import pandas as pd

csv_dir = "csv"
os.makedirs(csv_dir, exist_ok=True)

GREEN_POINTS_PATH = os.path.join(csv_dir, "green_points.csv")
STOCK_SYMBOLS_PATH = os.path.join(csv_dir, "stock_symbols.csv")
OUTPUT_PATH = os.path.join(csv_dir, "positions_filled.csv")

stock_df = pd.read_csv(STOCK_SYMBOLS_PATH)
green_df = pd.read_csv(GREEN_POINTS_PATH)

merged_df = pd.merge(green_df, stock_df, left_on='filename', right_on='file_saved', how='left')
merged_df.rename(columns={'x_label': 'date', 'y_value': 'position'}, inplace=True)

date_float = pd.to_numeric(merged_df['date'], errors='coerce')
year = date_float.fillna(0).astype(int)
month = np.floor((date_float - year) * 100 + 1e-6).astype(int)
month = month.clip(1, 12)
merged_df['date'] = year.astype(str) + '-' + month.astype(str).str.zfill(2)
merged_df['date'] = pd.to_datetime(merged_df['date'], format='%Y-%m', errors='coerce')
merged_df = merged_df.dropna(subset=['date', 'stock_name'])

def convert_position(value):
    if isinstance(value, str):
        value = value.strip().upper()
        if value.endswith('K'):
            return float(value[:-1]) * 1_000
        elif value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        elif value.endswith('B'):
            return float(value[:-1]) * 1_000_000_000
        else:
            try:
                return float(value)
            except:
                return np.nan
    return value

merged_df['position'] = merged_df['position'].apply(convert_position)

cols = ['stock_name'] + [col for col in merged_df.columns if col != 'stock_name']
merged_df = merged_df[cols].set_index('stock_name')
merged_df.drop(columns=['filename', 'file_saved', 'page'], inplace=True, errors='ignore')

merged_df = merged_df.reset_index()

all_filled = []
for stock, group in merged_df.groupby('stock_name'):
    group = group.sort_values('date')
    full_range = pd.date_range(start=group['date'].min(), end=group['date'].max(), freq='MS')
    filled = pd.DataFrame({'date': full_range})
    filled['stock_name'] = stock
    merged = pd.merge(filled, group, on=['stock_name', 'date'], how='left')
    merged['position'] = merged['position'].ffill()
    all_filled.append(merged)

filled_df = pd.concat(all_filled).sort_values(['stock_name', 'date'])
filled_df['date'] = filled_df['date'].dt.strftime('%Y-%m')
filled_df = filled_df[['stock_name', 'date', 'position']]
filled_df['year'] = filled_df['date'].str[:4].astype(int)
filled_df['month'] = filled_df['date'].str[5:].astype(int)
filled_df = filled_df[['stock_name', 'date', 'year', 'month', 'position']]

filled_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"Saved CSV : {OUTPUT_PATH}")
