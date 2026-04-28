import pandas as pd
import numpy as np

# Cleaning
df = pd.read_csv("CRSP_90_24_V2.csv")
df = df[df['SHRCD'].isin([10, 11, 12])].copy()
df['date'] = pd.to_datetime(df['date'])
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
df['DLRET'] = pd.to_numeric(df['DLRET'], errors='coerce')
sentinel_values = [-66.0, -77.0, -88.0, -99.0]
df.loc[df['RET'].isin(sentinel_values), 'RET'] = np.nan
mask = df['RET'].isna() & df['DLRET'].notna()
df.loc[mask, 'RET'] = df.loc[mask, 'DLRET']
df = df.dropna(subset=['RET'])
df['PRC'] = df['PRC'].abs()
df['MCAP'] = df['PRC'] * df['SHROUT']
df = df[df['MCAP'] > 0].copy()
df['SICCD'] = pd.to_numeric(df['SICCD'], errors='coerce')

print(f"CRSP rows loaded:{len(df):,}")
print(f"Unique stocks in CRSP:{df['PERMNO'].nunique():,}")
print(f"Stocks with SIC 9999:{(df['SICCD'] == 9999)['PERMNO' if False else df['SICCD'].eq(9999)].sum() if False else df[df['SICCD'] == 9999]['PERMNO'].nunique():,}")

# Load Compustat
comp = pd.read_csv("COMPSTAT_90_24.csv")
comp.columns = comp.columns.str.strip()
comp['sic'] = pd.to_numeric(comp['sic'], errors='coerce')
comp['datadate'] = pd.to_datetime(comp['datadate'], errors='coerce')

print(f"\nCompustat rows loaded: {len(comp):,}")
print(f"Unique tickers in Compustat:{comp['tic'].nunique():,}")

# Drop rows where SIC is missing or 9999 in Compustat
comp_clean = comp[comp['sic'].notna() & (comp['sic'] != 9999)].copy()
comp_clean = comp_clean.sort_values('datadate')

comp_lookup = (comp_clean.groupby('tic')['sic']
               .first()
               .reset_index()
               .rename(columns={'tic': 'TICKER', 'sic': 'SIC_from_compustat'}))

print(f"Compustat tickers with valid SIC: {len(comp_lookup):,}")

# Identify SIC 9999
crsp_9999_tickers = (df[df['SICCD'] == 9999]
                     .groupby('PERMNO')['TICKER']
                     .first()
                     .reset_index())

print(f"\nUnique PERMNOs with SIC 9999:  {len(crsp_9999_tickers):,}")

# Merge
patched = crsp_9999_tickers.merge(comp_lookup, on='TICKER', how='left')

matched = patched['SIC_from_compustat'].notna().sum()
unmatched = patched['SIC_from_compustat'].isna().sum()
print(f"Matched to Compustat SIC:      {matched:,}")
print(f"Still unmatched after merge:   {unmatched:,}  (remain as 9999)")

# Build PERMNO
permno_to_new_sic = (patched.dropna(subset=['SIC_from_compustat'])
                     .set_index('PERMNO')['SIC_from_compustat']
                     .astype(int)
                     .to_dict())

# Apply patch to CRSP dataframe
# Only overwrite rows where SICCD is currently 9999
mask_9999 = df['SICCD'] == 9999
df['SICCD_patched'] = df['SICCD'].copy()
df.loc[mask_9999, 'SICCD_patched'] = (
    df.loc[mask_9999, 'PERMNO'].map(permno_to_new_sic)
)
# Where no Compustat match was found, keep 9999
df['SICCD_patched'] = df['SICCD_patched'].fillna(df['SICCD'])

# Diagnostic
remaining_9999 = (df['SICCD_patched'] == 9999).sum()
print(f"\nRows still with SIC 9999 after patch: {remaining_9999:,}")
print(f"  (out of original {mask_9999.sum():,} rows with SIC 9999)")

# Save patched dataset
df.to_csv("CRSP_90_24_patched.csv", index=False)
print("\nPatched dataset saved to CRSP_90_24_patched.csv")
print("Use SICCD_patched instead of SICCD when assigning FF12 industries.")
print("\nColumn guide:")
print("  SICCD         = original CRSP SIC code (unchanged)")
print("  SICCD_patched = CRSP SIC, with 9999 replaced by Compustat SIC where available")