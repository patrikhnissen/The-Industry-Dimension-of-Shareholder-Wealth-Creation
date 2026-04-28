import pandas as pd
import numpy as np

# Cleaning
df = pd.read_csv("CRSP_90_24.csv")
df = df[df['SHRCD'].isin([10, 11, 12])].copy()
df['date']  = pd.to_datetime(df['date'])
df['RET']   = pd.to_numeric(df['RET'],   errors='coerce')
df['DLRET'] = pd.to_numeric(df['DLRET'], errors='coerce')
sentinel_values = [-66.0, -77.0, -88.0, -99.0]
df.loc[df['RET'].isin(sentinel_values), 'RET'] = np.nan
mask = df['RET'].isna() & df['DLRET'].notna()
df.loc[mask, 'RET'] = df.loc[mask, 'DLRET']
df = df.dropna(subset=['RET'])
df['PRC']  = df['PRC'].abs()
df['MCAP'] = df['PRC'] * df['SHROUT']
df = df[df['MCAP'] > 0].copy()

df['SICCD'] = pd.to_numeric(df['SICCD'], errors='coerce')

# SIC 9999 only
sic9999_df = df[df['SICCD'] == 9999].copy()

# Summarisw
summary = (sic9999_df.groupby('PERMNO')
           .agg(
               COMNAM   = ('COMNAM', 'last'),
               avg_mcap = ('MCAP',   'mean'),
               n_months = ('date',   'count'),
           )
           .reset_index()
           .sort_values('avg_mcap', ascending=False))

print("=" * 70)
print("TOP 50 STOCKS WITH SIC CODE 9999 BY AVERAGE MARKET CAP")
print("=" * 70)
print(f"\n{'PERMNO':>8}  {'Company Name':<35}  {'Avg MktCap ($k)':>16}  {'Months':>6}")
print("-" * 70)
for _, row in summary.head(50).iterrows():
    print(f"{int(row['PERMNO']):>8}  {str(row['COMNAM']):<35}  "
          f"{row['avg_mcap']:>16,.0f}  {int(row['n_months']):>6}")