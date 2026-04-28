import pandas as pd
import numpy as np

# Load and clean CRSP
df = pd.read_csv("CRSP_90_24_V2.csv")
df = df[df['SHRCD'].isin([10, 11, 12])].copy()
df['date']  = pd.to_datetime(df['date'])
df['RET']   = pd.to_numeric(df['RET'],   errors='coerce')
df['DLRET'] = pd.to_numeric(df['DLRET'], errors='coerce')

# Sentinel values
sentinel_values = [-66.0, -77.0, -88.0, -99.0]
df.loc[df['RET'].isin(sentinel_values), 'RET'] = np.nan

# Trin 1: Brug DLRET hvor RET mangler og DLRET er tilgængeligt
mask_dlret = df['RET'].isna() & df['DLRET'].notna()
df.loc[mask_dlret, 'RET'] = df.loc[mask_dlret, 'DLRET']

# Trin 2: Shumway imputation hvor både RET og DLRET mangler
# NYSE (EXCHCD=1) og AMEX (EXCHCD=2): -30%
# NASDAQ (EXCHCD=3): -55%
if 'EXCHCD' in df.columns:
    df['EXCHCD'] = pd.to_numeric(df['EXCHCD'], errors='coerce')
    mask_impute  = df['RET'].isna() & df['DLRET'].isna() & df['EXCHCD'].notna()
    n_nyse_amex  = (mask_impute & df['EXCHCD'].isin([1, 2])).sum()
    n_nasdaq     = (mask_impute & df['EXCHCD'].isin([3])).sum()
    df.loc[mask_impute & df['EXCHCD'].isin([1, 2]), 'RET'] = -0.30
    df.loc[mask_impute & df['EXCHCD'].isin([3]),    'RET'] = -0.55
    print(f"Shumway imputation: {n_nyse_amex:,} NYSE/AMEX obs sat til -30%, "
          f"{n_nasdaq:,} NASDAQ obs sat til -55%")
else:
    print("WARNING: EXCHCD ikke fundet i data. Shumway imputation sprunget over.")

# Nu droppes kun rækker hvor RET stadig mangler efter imputation
df = df.dropna(subset=['RET'])

df['PRC']  = df['PRC'].abs()
df['MCAP'] = df['PRC'] * df['SHROUT']
df = df[df['MCAP'] > 0].copy()
df['SICCD'] = pd.to_numeric(df['SICCD'], errors='coerce')

print(f"CRSP rows loaded:          {len(df):,}")
print(f"Unique stocks in CRSP:     {df['PERMNO'].nunique():,}")
print(f"Stocks with SIC 9999:      {df[df['SICCD'] == 9999]['PERMNO'].nunique():,}")

# Load Compustat
comp = pd.read_csv("COMPSTAT_90_24.csv")
comp.columns = comp.columns.str.strip()
comp['sic']      = pd.to_numeric(comp['sic'],      errors='coerce')
comp['datadate'] = pd.to_datetime(comp['datadate'], errors='coerce')

print(f"\nCompustat rows loaded:     {len(comp):,}")
print(f"Unique tickers in Compustat: {comp['tic'].nunique():,}")

# Drop rows where SIC is missing or 9999 in Compustat
comp_clean = comp[comp['sic'].notna() & (comp['sic'] != 9999)].copy()
comp_clean = comp_clean.sort_values('datadate')

comp_lookup = (comp_clean.groupby('tic')['sic']
               .first()
               .reset_index()
               .rename(columns={'tic': 'TICKER', 'sic': 'SIC_from_compustat'}))

print(f"Compustat tickers with valid SIC: {len(comp_lookup):,}")

# Identify CRSP stocks with SIC 9999
crsp_9999_tickers = (df[df['SICCD'] == 9999]
                     .groupby('PERMNO')['TICKER']
                     .first()
                     .reset_index())

print(f"\nUnique PERMNOs with SIC 9999: {len(crsp_9999_tickers):,}")

# Merge to get Compustat SIC for 9999 stocks
patched   = crsp_9999_tickers.merge(comp_lookup, on='TICKER', how='left')
matched   = patched['SIC_from_compustat'].notna().sum()
unmatched = patched['SIC_from_compustat'].isna().sum()
print(f"Matched to Compustat SIC:     {matched:,}")
print(f"Still unmatched after merge:  {unmatched:,}  (forbliver 9999)")

# Build PERMNO -> new SIC mapping
permno_to_new_sic = (patched.dropna(subset=['SIC_from_compustat'])
                     .set_index('PERMNO')['SIC_from_compustat']
                     .astype(int)
                     .to_dict())

# Apply patch to CRSP dataframe
mask_9999 = df['SICCD'] == 9999
df['SICCD_patched'] = df['SICCD'].copy()
df.loc[mask_9999, 'SICCD_patched'] = (
    df.loc[mask_9999, 'PERMNO'].map(permno_to_new_sic)
)
df['SICCD_patched'] = df['SICCD_patched'].fillna(df['SICCD'])

remaining_9999 = (df['SICCD_patched'] == 9999).sum()
print(f"\nRows stadig med SIC 9999 efter patch: {remaining_9999:,}")
print(f"  (ud af originale {mask_9999.sum():,} rækker med SIC 9999)")

# Save patched dataset
df.to_csv("CRSP_90_24_patched_shumway.csv", index=False)
print("\nPatched dataset gemt som CRSP_90_24_patched.csv")
print("Brug SICCD_patched i stedet for SICCD ved FF12-tildeling.")
print("\nKolonne-guide:")
print("  SICCD         = original CRSP SIC kode (uændret)")
print("  SICCD_patched = CRSP SIC, med 9999 erstattet af Compustat SIC hvor tilgængeligt")
print("  RET           = inkluderer Shumway imputation (-30% NYSE/AMEX, -55% NASDAQ)")