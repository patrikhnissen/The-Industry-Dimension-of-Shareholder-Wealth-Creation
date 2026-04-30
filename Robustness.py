import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# FF12 Industry Classification
def assign_ff12(sic):
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


def add_industry(df):
    df = df.copy()
    df['SICCD_patched'] = pd.to_numeric(df['SICCD_patched'], errors='coerce')
    df['FF12_timevary'] = df['SICCD_patched'].apply(assign_ff12)
    return df


def apply_entry_sic(df):
    entry_industry = (df.sort_values(['PERMNO', 'date'])
                        .groupby('PERMNO')['FF12_timevary']
                        .first()
                        .reset_index()
                        .rename(columns={'FF12_timevary': 'FF12'}))
    df = df.merge(entry_industry, on='PERMNO', how='left')
    return df


def compute_wealth_creation(df):
    df = df.sort_values(['PERMNO', 'date'])

    vw_monthly = df.groupby('date').apply(
        lambda g: np.average(g['RET'], weights=g['MCAP']))
    market_gross  = float((1 + vw_monthly).prod())
    market_return = market_gross - 1

    records = []
    for permno, group in df.groupby('PERMNO'):
        group = group.sort_values('date').reset_index(drop=True)

        gross_stock = (1 + group['RET']).prod()
        gross_tbill = (1 + group['RF']).prod()
        bh_return   = gross_stock - 1

        if bh_return < -1:
            bh_return   = -1.0
            gross_stock =  0.0

        tbill_return = gross_tbill - 1
        me_0         = group['MCAP'].iloc[0]
        industry     = group['FF12'].iloc[0] if 'FF12' in group.columns \
                           else 'Unknown'

        rf_gross      = 1 + group['RF']
        total_rf_prod = rf_gross.prod()
        cum_rf_to_t   = rf_gross.cumprod()
        fv_forward    = total_rf_prod / cum_rf_to_t

        mcap_lag   = group['MCAP'].shift(1).fillna(group['MCAP'].iloc[0])
        excess_ret = group['RET'] - group['RF']
        wc         = (mcap_lag * excess_ret * fv_forward).sum()

        records.append({
            'PERMNO':          permno,
            'FF12':            industry,
            'n_months':        len(group),
            'bh_return':       bh_return,
            'tbill_return':    tbill_return,
            'beat_tbill':      int(bh_return > tbill_return),
            'beat_market':     int(bh_return > market_return),
            'positive':        int(bh_return > 0),
            'me_0':            me_0,
            'wealth_creation': wc,
            'positive_wc':     int(wc > 0),
        })

    stock_df = pd.DataFrame(records)

    n_total    = len(stock_df)
    n_positive = stock_df['positive'].sum()
    n_beat_tb  = stock_df['beat_tbill'].sum()
    n_beat_mkt = stock_df['beat_market'].sum()
    n_clipped  = (stock_df['bh_return'] == -1.0).sum()

    print("\n" + "="*65)
    print("AGGREGATE RETURN STATISTICS (full sample 1990-2024)")
    print("="*65)
    print(f"Total unique stocks:              {n_total:>8,}")
    print(f"Stocks with positive BHR:         {n_positive:>8,}  ({n_positive/n_total:.1%})")
    print(f"Stocks beating T-bills:           {n_beat_tb:>8,}  ({n_beat_tb/n_total:.1%})")
    print(f"Stocks beating VW market:         {n_beat_mkt:>8,}  ({n_beat_mkt/n_total:.1%})")
    print(f"Mean lifetime BHR:                {stock_df['bh_return'].mean():>8.2%}")
    print(f"Median lifetime BHR:              {stock_df['bh_return'].median():>8.2%}")
    print(f"Skewness of lifetime BHR:         {stock_df['bh_return'].skew():>8.2f}")
    print(f"BHR observations clipped to -100%:{n_clipped:>8,}")
    print("="*65)

    return stock_df


# SHAREHOLDER WEALTH CREATION BY FAMA-FRENCH 12 INDUSTRY
FF12_ORDER = [
    'Consumer Non-Durables', 'Consumer Durables', 'Manufacturing',
    'Energy', 'Chemicals', 'Business Equipment', 'Telecoms',
    'Utilities', 'Shops', 'Healthcare', 'Finance', 'Other'
]

def table6_bessembinder_style(stock_df, df_monthly):
    total_wc   = stock_df['wealth_creation'].sum()
    total_wc_m = total_wc / 1000
    total_fm   = len(df_monthly)

    rows = []
    for ind in FF12_ORDER:
        sub    = stock_df[stock_df['FF12'] == ind]
        sub_fm = df_monthly[df_monthly['FF12'] == ind]
        if len(sub) == 0:
            continue

        ind_wc   = sub['wealth_creation'].sum()
        ind_wc_m = ind_wc / 1000
        n_fm     = len(sub_fm)
        wc_share = ind_wc / total_wc if total_wc != 0 else np.nan
        fm_share = n_fm / total_fm   if total_fm != 0 else np.nan
        ratio    = wc_share / fm_share if fm_share != 0 else np.nan

        rows.append({
            'Industry':     ind,
            'WC ($m)':      ind_wc_m,
            'Firm/Months':  n_fm,
            'WC%':          wc_share,
            'Firm/Months%': fm_share,
            'Ratio WC/FM':  ratio,
        })

    summary = pd.DataFrame(rows)

    print("\n" + "="*100)
    print("SHAREHOLDER WEALTH CREATION BY FAMA-FRENCH 12 INDUSTRY")
    print("="*100)
    print(f"{'Industry':<26} {'WC ($m)':>14}  {'Firm/Months':>12}  "
          f"{'WC%':>7}  {'FM%':>7}  {'Ratio':>7}")
    print("-"*100)
    for _, row in summary.iterrows():
        print(f"{row['Industry']:<26} "
              f"{row['WC ($m)']:>14,.0f}  "
              f"{int(row['Firm/Months']):>12,}  "
              f"{row['WC%']:>7.2%}  "
              f"{row['Firm/Months%']:>7.2%}  "
              f"{row['Ratio WC/FM']:>7.2f}")
    print("-"*100)
    print(f"{'TOTAL':<26} "
          f"{total_wc_m:>14,.0f}  "
          f"{total_fm:>12,}  "
          f"{'100.00%':>7}  "
          f"{'100.00%':>7}  "
          f"{'1.00':>7}")
    print("="*100)

    summary.to_csv("table_robust_style_90_24.csv", index=False)
    print("Saved: table2_industry_wc_bessembinder_style_90_24.csv")

    return summary

# WITHIN-INDUSTRY CONCENTRATION
def table7_within_industry_concentration(stock_df):
    rows = []
    for ind in FF12_ORDER:
        sub = stock_df[stock_df['FF12'] == ind].copy()
        if len(sub) < 10:
            continue

        sub_sorted = sub.sort_values('wealth_creation', ascending=False)
        gross_wc   = sub[sub['wealth_creation'] > 0]['wealth_creation'].sum()
        net_wc     = sub['wealth_creation'].sum()

        def top_share(pct):
            n_top  = max(1, int(np.ceil(len(sub) * pct)))
            top_wc = sub_sorted.head(n_top)['wealth_creation'].sum()
            return top_wc / gross_wc if gross_wc != 0 else np.nan

        cum_wc     = sub_sorted['wealth_creation'].cumsum()
        n_for_half = (cum_wc < gross_wc * 0.5).sum() + 1

        if net_wc > 0:
            n_for_net   = int((cum_wc < net_wc).sum() + 1)
            pct_for_net = n_for_net / len(sub)
        else:
            n_for_net   = np.nan
            pct_for_net = np.nan

        rows.append({
            'Industry':         ind,
            'N':                len(sub),
            'Top 1% WC Share':  top_share(0.01),
            'Top 5% WC Share':  top_share(0.05),
            'Top 10% WC Share': top_share(0.10),
            'N for 50% WC':     n_for_half,
            '% for 50% WC':     n_for_half / len(sub),
            'N for net WC':     n_for_net,
            '% for net WC':     pct_for_net,
        })

    result = pd.DataFrame(rows)

    print("\n" + "="*115)
    print("WITHIN-INDUSTRY WEALTH CREATION CONCENTRATION")
    print("="*115)
    print(f"{'Industry':<26} {'N':>6}  {'Top1%':>8}  {'Top5%':>8}  "
          f"{'Top10%':>8}  {'N@50%':>7}  {'%@50%':>7}  {'N@net':>7}  {'%@net':>7}")
    print("-"*115)
    for _, row in result.iterrows():
        n_net   = int(row['N for net WC']) if not np.isnan(row['N for net WC']) else 'N/A'
        pct_net = f"{row['% for net WC']:.2%}" if not np.isnan(row['% for net WC']) else 'N/A'
        print(f"{row['Industry']:<26} {int(row['N']):>6}  "
              f"{row['Top 1% WC Share']:>8.2%}  "
              f"{row['Top 5% WC Share']:>8.2%}  "
              f"{row['Top 10% WC Share']:>8.2%}  "
              f"{int(row['N for 50% WC']):>7}  "
              f"{row['% for 50% WC']:>7.2%}  "
              f"{str(n_net):>7}  "
              f"{pct_net:>7}")
    print("="*115)

    result.to_csv("table_within_industry_concentration_robust_90_24.csv", index=False)
    print("Saved: table2_within_industry_concentration_90_24.csv")

    return result

# RETURN DISTRIBUTION BY INDUSTRY
def table8_return_distribution(stock_df):
    rows = []
    for ind in FF12_ORDER:
        sub = stock_df[stock_df['FF12'] == ind]
        if len(sub) == 0:
            continue
        rows.append({
            'Industry':      ind,
            'N':             len(sub),
            'Mean':          sub['bh_return'].mean(),
            'Median':        sub['bh_return'].median(),
            'Skewness':      sub['bh_return'].skew(),
            '% Positive':    sub['positive'].mean(),
            '% > T-bill':    sub['beat_tbill'].mean(),
            '% > VW Market': sub['beat_market'].mean(),
        })

    result = pd.DataFrame(rows)

    print("\n" + "="*95)
    print("TABLE 8: RETURN DISTRIBUTION BY INDUSTRY")
    print("="*95)
    print(f"{'Industry':<26} {'N':>6}  {'Mean':>10}  {'Median':>10}  "
          f"{'Skew':>7}  {'%>0':>7}  {'%>Tbill':>8}  {'%>Mkt':>7}")
    print("-"*95)
    for _, row in result.iterrows():
        print(f"{row['Industry']:<26} {int(row['N']):>6}  "
              f"{row['Mean']:>10.2%}  "
              f"{row['Median']:>10.2%}  "
              f"{row['Skewness']:>7.2f}  "
              f"{row['% Positive']:>7.2%}  "
              f"{row['% > T-bill']:>8.2%}  "
              f"{row['% > VW Market']:>7.2%}")
    print("-"*95)
    total = stock_df
    print(f"{'TOTAL':<26} {len(total):>6}  "
          f"{total['bh_return'].mean():>10.2%}  "
          f"{total['bh_return'].median():>10.2%}  "
          f"{total['bh_return'].skew():>7.2f}  "
          f"{total['positive'].mean():>7.2%}  "
          f"{total['beat_tbill'].mean():>8.2%}  "
          f"{total['beat_market'].mean():>7.2%}")
    print("="*95)

    result.to_csv("table_return_distribution_robust_90_24.csv", index=False)
    print("Saved: table_return_distribution_90_24.csv")

    return result

# FIGURE: RETURN DISTRIBUTION (FULL SAMPLE)
def figure2_return_distributions(stock_df):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#d4d4d4',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.7,
    })

    BLUE = '#2166ac'

    CLIP_LOW = -1.0
    CLIP_HIGH = 10.0

    bhr = stock_df['bh_return'].dropna()
    bhr_plot = bhr.clip(lower=CLIP_LOW, upper=CLIP_HIGH)
    n_clipped = (bhr > CLIP_HIGH).sum()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(bhr_plot, bins=200, color=BLUE, alpha=0.7, edgecolor='none')

    ax.axvline(x=0, color='black', linewidth=0.7, alpha=0.4)

    tick_vals = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f'{int(v * 100)}%' for v in tick_vals], fontsize=9)
    ax.set_xlim(CLIP_LOW, CLIP_HIGH)
    ax.tick_params(axis='x', colors='#2c2c2c', labelsize=9)
    ax.tick_params(axis='y', colors='#2c2c2c', labelsize=9)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(500))

    ax.set_xlabel('Lifetime buy-and-hold return', fontsize=10, color='#2c2c2c')
    ax.set_ylabel('Number of observations', fontsize=10, color='#2c2c2c')

    ax.text(0.98, 0.97, f'N={len(bhr):,}  ({n_clipped} clipped)',
            transform=ax.transAxes, fontsize=8,
            ha='right', va='top', color='#2c2c2c')

    ax.set_title(
        'Distribution of Buy-and-Hold Returns\n'
        'Full sample 1990-2024.',
        fontsize=11, color='#2c2c2c', pad=12
    )

    plt.tight_layout()
    fig.savefig("figure_return_distributions_robust.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: figure_return_distributions.png")

# MAIN
if __name__ == '__main__':

    df = pd.read_csv("CRSP_90_24_patched_shumway.csv")
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

    rows_before = len(df)
    df = df.sort_values(['PERMNO', 'date', 'RET'], na_position='last')
    df = df.drop_duplicates(subset=['PERMNO', 'date'], keep='first')
    rows_after = len(df)
    print(f"Deduplication: removed {rows_before - rows_after:,} duplicate "
          f"firm-month observations ({rows_before:,} -> {rows_after:,} rows)")

    ff = pd.read_csv("F-F_Research_Data_Factors.CSV", skiprows=3)
    ff = ff[ff.iloc[:, 0].astype(str).str.match(r'^\s*\d{6}\s*$', na=False)].copy()
    ff.columns = ff.columns.str.strip()
    ff = ff.rename(columns={ff.columns[0]: 'date_raw'})
    ff['date_ff'] = (pd.to_datetime(ff['date_raw'].str.strip(), format='%Y%m')
                     + pd.offsets.MonthEnd(0))
    ff['RF'] = pd.to_numeric(ff['RF'], errors='coerce') / 100

    df['date_month'] = df['date'] + pd.offsets.MonthEnd(0)
    df = df.merge(ff[['date_ff', 'RF']], left_on='date_month',
                  right_on='date_ff', how='left')
    df = df.drop(columns=['date_month', 'date_ff'])

    n_missing_rf = df['RF'].isna().sum()
    if n_missing_rf > 0:
        print(f"WARNING: {n_missing_rf} rows have no RF match.")
    else:
        print("RF merged successfully.")

    df = add_industry(df)
    df = apply_entry_sic(df)

    print("\nComputing per-stock wealth creation and lifetime returns...")
    stock_df = compute_wealth_creation(df)

    t6 = table6_bessembinder_style(stock_df, df)
    table7_within_industry_concentration(stock_df)
    table8_return_distribution(stock_df)
    figure2_return_distributions(stock_df)

    print("\nDone. Outputs:")
    print("  table2_industry_wc_bessembinder_style_90_24.csv")
    print("  table2_within_industry_concentration_90_24.csv")
    print("  table2_return_distribution_90_24.csv")
    print("  figure2_return_distributions.png")
    print("  figure2_lottery_vs_yield.png")
    print("  figure2_appendix_all_industries_wc.png")