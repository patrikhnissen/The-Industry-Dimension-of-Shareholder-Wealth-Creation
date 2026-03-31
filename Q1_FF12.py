import pandas as pd
import numpy as np

# =============================================================================
# FAMA-FRENCH 12 INDUSTRY CLASSIFICATION
# =============================================================================

def assign_ff12(sic):
    """
    Assigns a SIC code to one of the Fama-French 12 industry groups.
    Returns the industry name as a string.
    NaN / missing SIC codes return 'Other'.
    """
    try:
        sic = int(sic)
    except (ValueError, TypeError):
        return 'Other'

    if (100 <= sic <= 999) or (2000 <= sic <= 2399) or \
       (2700 <= sic <= 2749) or (2770 <= sic <= 2799) or \
       (3100 <= sic <= 3199) or (3940 <= sic <= 3989):
        return 'Consumer Non-Durables'

    elif (2500 <= sic <= 2519) or (2590 <= sic <= 2599) or \
         (3630 <= sic <= 3659) or (3710 <= sic <= 3711) or \
         (3714 <= sic <= 3714) or (3716 <= sic <= 3716) or \
         (3750 <= sic <= 3751) or (3792 <= sic <= 3792) or \
         (3900 <= sic <= 3939) or (3990 <= sic <= 3999):
        return 'Consumer Durables'

    elif (2520 <= sic <= 2589) or (2600 <= sic <= 2699) or \
         (2750 <= sic <= 2769) or (3000 <= sic <= 3099) or \
         (3200 <= sic <= 3569) or (3580 <= sic <= 3629) or \
         (3700 <= sic <= 3709) or (3712 <= sic <= 3713) or \
         (3715 <= sic <= 3715) or (3717 <= sic <= 3749) or \
         (3752 <= sic <= 3791) or (3793 <= sic <= 3799) or \
         (3830 <= sic <= 3839) or (3860 <= sic <= 3899):
        return 'Manufacturing'

    elif (1200 <= sic <= 1399) or (2900 <= sic <= 2999):
        return 'Energy'

    elif (2800 <= sic <= 2829) or (2840 <= sic <= 2899):
        return 'Chemicals'

    elif (3570 <= sic <= 3579) or \
         (3660 <= sic <= 3692) or (3694 <= sic <= 3699) or \
         (3810 <= sic <= 3829) or \
         (7370 <= sic <= 7379):
        return 'Business Equipment'

    elif 4800 <= sic <= 4899:
        return 'Telecoms'

    elif (4900 <= sic <= 4949) or (4950 <= sic <= 4991):
        return 'Utilities'

    elif (5000 <= sic <= 5999) or (7200 <= sic <= 7299) or \
         (7600 <= sic <= 7699):
        return 'Shops'

    elif (2830 <= sic <= 2836) or \
         (3693 <= sic <= 3693) or \
         (3840 <= sic <= 3859) or \
         (8000 <= sic <= 8099):
        return 'Healthcare'

    elif 6000 <= sic <= 6999:
        return 'Finance'

    else:
        return 'Other'

# =============================================================================
# APPLY TO DATAFRAME
# =============================================================================

def add_industry(df):
    """Adds FF12 industry label to each row based on SICCD."""
    df = df.copy()
    df['SICCD_patched'] = pd.to_numeric(df['SICCD_patched'], errors='coerce')
    df['FF12'] = df['SICCD_patched'].apply(assign_ff12)
    return df

# =============================================================================
# LIFETIME RETURN ANALYSIS BY INDUSTRY
# =============================================================================

def compute_lifetime_returns(df):
    """
    For each unique stock (PERMNO), computes:
      - Lifetime buy-and-hold return (linking monthly gross returns)
      - Matched cumulative T-bill return over same months
      - Whether the stock beat T-bill over its lifetime
      - Industry group (FF12)

    Returns a DataFrame with one row per stock.
    """
    df = df.sort_values(['PERMNO', 'date'])

    records = []
    for permno, group in df.groupby('PERMNO'):
        gross_stock = (1 + group['RET']).prod()
        gross_tbill = (1 + group['RF']).prod()

        bh_return    = gross_stock - 1
        tbill_return = gross_tbill - 1

        industry = group['FF12'].mode().iloc[0] if 'FF12' in group.columns \
                   else 'Unknown'

        records.append({
            'PERMNO':       permno,
            'FF12':         industry,
            'n_months':     len(group),
            'bh_return':    bh_return,
            'tbill_return': tbill_return,
            'beat_tbill':   int(bh_return > tbill_return),
            'positive':     int(bh_return > 0),
        })

    return pd.DataFrame(records)

def industry_summary(lifetime_df):
    """
    Summarises lifetime returns by FF12 industry.
    """
    ff12_order = [
        'Consumer Non-Durables', 'Consumer Durables', 'Manufacturing',
        'Energy', 'Chemicals', 'Business Equipment', 'Telecoms',
        'Utilities', 'Shops', 'Healthcare', 'Finance', 'Other'
    ]

    rows = []
    for ind in ff12_order:
        sub = lifetime_df[lifetime_df['FF12'] == ind]
        if len(sub) == 0:
            continue
        rows.append({
            'Industry':      ind,
            'N':             len(sub),
            'Mean BH Ret':   sub['bh_return'].mean(),
            'Median BH Ret': sub['bh_return'].median(),
            'Skewness':      sub['bh_return'].skew(),
            '% Positive':    sub['positive'].mean(),
            '% > T-bill':    sub['beat_tbill'].mean(),
        })

    summary = pd.DataFrame(rows)

    print("\n" + "="*85)
    print("LIFETIME BUY-AND-HOLD RETURNS BY FAMA-FRENCH 12 INDUSTRY")
    print("="*85)
    print(f"{'Industry':<26} {'N':>6}  {'Mean':>10}  {'Median':>10}  "
          f"{'Skew':>7}  {'%>0':>8}  {'%>Tbill':>9}")
    print("-"*85)
    for _, row in summary.iterrows():
        print(f"{row['Industry']:<26} {int(row['N']):>6}  "
              f"{row['Mean BH Ret']:>10.2%}  {row['Median BH Ret']:>10.2%}  "
              f"{row['Skewness']:>7.2f}  {row['% Positive']:>8.2%}  "
              f"{row['% > T-bill']:>9.2%}")
    print("-"*85)
    total = lifetime_df
    print(f"{'TOTAL':<26} {len(total):>6}  "
          f"{total['bh_return'].mean():>10.2%}  "
          f"{total['bh_return'].median():>10.2%}  "
          f"{total['bh_return'].skew():>7.2f}  "
          f"{total['positive'].mean():>8.2%}  "
          f"{total['beat_tbill'].mean():>9.2%}")
    print("="*85)

    return summary

# =============================================================================
# BOOTSTRAP BY INDUSTRY
# =============================================================================

def build_monthly_data_for_industry(df, industry):
    """
    Builds a monthly_data dict filtered to one FF12 industry.
    """
    sub = df[df['FF12'] == industry]
    monthly_data = {}
    for month, group in sub.groupby('date'):
        if len(group) == 0:
            continue
        monthly_data[month] = {
            'ret':   group['RET'].values.astype(np.float64),
            'mcap':  group['MCAP'].values.astype(np.float64),
            'tbill': float(group['RF'].iloc[0]),
        }
    return monthly_data

def run_industry_bootstrap(df, n_simulations=5000, seed=42):
    """
    Runs single-stock bootstrap for each FF12 industry separately.

    Returns a DataFrame summarising results by industry.
    """
    ff12_industries = [
        'Consumer Non-Durables', 'Consumer Durables', 'Manufacturing',
        'Energy', 'Chemicals', 'Business Equipment', 'Telecoms',
        'Utilities', 'Shops', 'Healthcare', 'Finance', 'Other'
    ]

    rng = np.random.default_rng(seed)
    results = []

    for industry in ff12_industries:
        monthly_ind = build_monthly_data_for_industry(df, industry)
        if len(monthly_ind) < 12:
            print(f"  Skipping {industry} — too few months of data")
            continue

        months = sorted(monthly_ind.keys())
        stock_rets = np.empty(n_simulations)
        tbill_rets = np.empty(n_simulations)

        for i in range(n_simulations):
            gross_stock = 1.0
            gross_tbill = 1.0
            for month in months:
                ret  = monthly_ind[month]['ret']
                rf   = monthly_ind[month]['tbill']
                idx  = rng.integers(0, len(ret))
                gross_stock *= (1 + ret[idx])
                gross_tbill *= (1 + rf)

            stock_rets[i] = gross_stock - 1
            tbill_rets[i] = gross_tbill - 1

        s = pd.Series(stock_rets)
        t = pd.Series(tbill_rets)

        results.append({
            'Industry':   industry,
            'N_months':   len(months),
            'Mean':       s.mean(),
            'Median':     s.median(),
            'Skewness':   s.skew(),
            '% > 0':      (s > 0).mean(),
            '% > T-bill': (s > t).mean(),
        })


        print(f"  {industry:<26} done — "
              f"Median: {s.median():.1%}, %>T-bill: {(s>t).mean():.1%}")

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # --- Load and clean data ---
    df = pd.read_csv("CRSP_90_24_patched.csv")
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

    # --- Load Kenneth French risk-free rate (1-month T-bill) ---
    ff = pd.read_csv("F-F_Research_Data_Factors.CSV", skiprows=3)

    # Keep only rows whose first column is a 6-digit YYYYMM date string.
    ff = ff[ff.iloc[:, 0].astype(str).str.match(r'^\s*\d{6}\s*$', na=False)].copy()
    ff.columns = ff.columns.str.strip()
    ff = ff.rename(columns={ff.columns[0]: 'date_raw'})
    ff['date_ff'] = (pd.to_datetime(ff['date_raw'].str.strip(), format='%Y%m')
                     + pd.offsets.MonthEnd(0))
    ff['RF'] = pd.to_numeric(ff['RF'], errors='coerce') / 100  # % → decimal

    # Align CRSP dates to month-end so the merge keys match
    df['date_month'] = df['date'] + pd.offsets.MonthEnd(0)
    df = df.merge(ff[['date_ff', 'RF']], left_on='date_month', right_on='date_ff',
                  how='left')
    df = df.drop(columns=['date_month', 'date_ff'])

    n_missing_rf = df['RF'].isna().sum()
    if n_missing_rf > 0:
        print(f"WARNING: {n_missing_rf} rows have no RF match — check date coverage.")
    else:
        print("RF merged successfully from F-F_Research_Data_Factors.CSV")

    # --- Step 1: Assign FF12 industry labels ---
    df = add_industry(df)
    print("\nFF12 industry distribution:")
    print(df.groupby('FF12')['PERMNO'].nunique().sort_values(ascending=False)
            .rename('Unique stocks').to_string())

    # --- Step 2: Lifetime returns by industry ---
    print("\nComputing lifetime returns per stock...")
    lifetime_df = compute_lifetime_returns(df)
    summary = industry_summary(lifetime_df)

    # --- Step 3: Bootstrap single-stock simulation by industry ---
    print("\nRunning single-stock bootstrap by industry (5,000 simulations each)...")
    bootstrap_by_industry = run_industry_bootstrap(df, n_simulations=5000)

    print("\n" + "="*75)
    print("SINGLE-STOCK BOOTSTRAP RESULTS BY INDUSTRY")
    print("="*75)
    # BUG FIX 4 (continued): corrected column key from ' % > 0' to '% > 0'
    print(f"{'Industry':<26} {'Mean':>10}  {'Median':>10}  "
          f"{'Skew':>8}  {'%>0':>8}  {'%>Tbill':>9}")
    print("-"*75)
    for _, row in bootstrap_by_industry.iterrows():
        print(f"{row['Industry']:<26} {row['Mean']:>10.2%}  "
              f"{row['Median']:>10.2%}  {row['Skewness']:>8.2f}  "
              f"{row['% > 0']:>8.2%}  {row['% > T-bill']:>9.2%}")
    print("="*75)

    # --- Save results ---
    summary.to_csv("industry_lifetime_summary_90_24.csv", index=False)
    bootstrap_by_industry.to_csv("industry_bootstrap_summary_90_24.csv", index=False)
    print("\nResults saved to industry_lifetime_summary_90_24.csv "
          "and industry_bootstrap_summary_90_24.csv")
