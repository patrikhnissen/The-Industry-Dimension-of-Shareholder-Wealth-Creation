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
         (3810 <= sic <= 3829) or (7370 <= sic <= 7379):
        return 'Business Equipment'
    elif 4800 <= sic <= 4899:
        return 'Telecoms'
    elif (4900 <= sic <= 4949) or (4950 <= sic <= 4991):
        return 'Utilities'
    elif (5000 <= sic <= 5999) or (7200 <= sic <= 7299) or \
         (7600 <= sic <= 7699):
        return 'Shops'
    elif (2830 <= sic <= 2836) or (3693 <= sic <= 3693) or \
         (3840 <= sic <= 3859) or (8000 <= sic <= 8099):
        return 'Healthcare'
    elif 6000 <= sic <= 6999:
        return 'Finance'
    else:
        return 'Other'


def add_industry(df):
    df = df.copy()
    df['SICCD_patched'] = pd.to_numeric(df['SICCD_patched'], errors='coerce')
    df['FF12'] = df['SICCD_patched'].apply(assign_ff12)
    return df

# BUILD STOCK UNIVERSE
def build_stock_universe(df, market_monthly):

    df = df.sort_values(['PERMNO', 'date'])
    universe = {}
    for permno, group in df.groupby('PERMNO'):
        dates    = group['date'].dt.to_period('M')
        mkt_rets = np.array([
            market_monthly.get(d, 0.0) for d in dates
        ], dtype=np.float64)
        universe[permno] = {
            'rets':     group['RET'].values.astype(np.float64),
            'rf':       group['RF'].values.astype(np.float64),
            'mkt':      mkt_rets,
            'industry': group['FF12'].iloc[0],
        }
    return universe

# HORIZON-BASED BOOTSTRAP (unconstrained)
def bootstrap_horizon(universe, permnos, portfolio_sizes, n_sims, rng,
                      horizon_months, market_full_gross, tbill_full_gross,
                      full_sample=False):

    results = []

    for n in portfolio_sizes:
        port_rets   = np.empty(n_sims)
        tbill_rets  = np.empty(n_sims)
        market_rets = np.empty(n_sims)

        for i in range(n_sims):
            drawn_idx     = rng.choice(len(permnos), size=n, replace=True)
            stock_grosses = []
            tbill_grosses = []
            mkt_grosses   = []

            for idx in drawn_idx:
                info = universe[permnos[idx]]
                rets = info['rets']
                rf   = info['rf']
                mkt  = info['mkt']
                m    = len(rets)

                if full_sample:
                    w_rets = rets
                    w_rf   = rf
                    w_mkt  = mkt
                else:
                    if m == 0:
                        stock_grosses.append(1.0)
                        tbill_grosses.append(1.0)
                        mkt_grosses.append(1.0)
                        continue
                    start  = rng.integers(0, m)
                    end    = min(start + horizon_months, m)
                    w_rets = rets[start:end]
                    w_rf   = rf[start:end]
                    w_mkt  = mkt[start:end]

                stock_grosses.append(float(np.prod(1 + w_rets)))
                tbill_grosses.append(float(np.prod(1 + w_rf)))
                mkt_grosses.append(float(np.prod(1 + w_mkt)))

            port_rets[i]   = float(np.mean(stock_grosses)) - 1
            tbill_rets[i]  = float(np.mean(tbill_grosses)) - 1
            market_rets[i] = float(np.mean(mkt_grosses))   - 1

        s = pd.Series(port_rets)

        if full_sample:
            tbill_bench  = tbill_full_gross - 1
            market_bench = market_full_gross - 1
            pct_tbill    = (s > tbill_bench).mean()
            pct_market   = (s > market_bench).mean()
        else:
            pct_tbill  = (port_rets > tbill_rets).mean()
            pct_market = (port_rets > market_rets).mean()

        results.append({
            'n':             n,
            'Mean':          s.mean(),
            'Median':        s.median(),
            'Skew':          s.skew(),
            '% > 0':         (s > 0).mean(),
            '% > T-bill':    pct_tbill,
            '% > VW Market': pct_market,
        })

    return pd.DataFrame(results)

# INDUSTRY-STRATIFIED BOOTSTRAP (full sample)
FF12_INDUSTRIES = [
    'Consumer Non-Durables', 'Consumer Durables', 'Manufacturing',
    'Energy', 'Chemicals', 'Business Equipment', 'Telecoms',
    'Utilities', 'Shops', 'Healthcare', 'Finance', 'Other'
]

def bootstrap_stratified_fullsample(universe, n_sims, rng,
                                    tbill_full_ret, market_full_ret):

    ind_permnos = {ind: [] for ind in FF12_INDUSTRIES}
    for permno, info in universe.items():
        ind = info['industry']
        if ind in ind_permnos:
            ind_permnos[ind].append(permno)

    available = [i for i in FF12_INDUSTRIES if len(ind_permnos[i]) > 0]
    n_avail   = len(available)
    results   = []

    for k in range(1, n_avail + 1):
        port_rets = []

        for _ in range(n_sims):
            chosen  = rng.choice(n_avail, size=k, replace=False)
            grosses = []
            for ci in chosen:
                ind    = available[ci]
                pool   = ind_permnos[ind]
                permno = pool[rng.integers(0, len(pool))]
                rets   = universe[permno]['rets']
                grosses.append(float(np.prod(1 + rets)))
            port_rets.append(float(np.mean(grosses)) - 1)

        s = pd.Series(port_rets)
        results.append({
            'K':             k,
            'Mean':          s.mean(),
            'Median':        s.median(),
            'Skew':          s.skew(),
            '% > 0':         (s > 0).mean(),
            '% > T-bill':    (s > tbill_full_ret).mean(),
            '% > VW Market': (s > market_full_ret).mean(),
        })
        print(f"  K={k:>2}  %>T-bill: {(s > tbill_full_ret).mean():.1%}  "
              f"  %>Market: {(s > market_full_ret).mean():.1%}")

    return pd.DataFrame(results)

# BOOTSTRAP TABLES
def print_table11(results_by_horizon):
    horizons = list(results_by_horizon.keys())

    print("\n" + "="*115)
    print("BOOTSTRAP RESULTS — UNCONSTRAINED RANDOM PORTFOLIOS")
    print("Replicating Bessembinder (2018) Table 4 | Sample: 1990-2024")
    print("Note: Market benchmark matched per simulation to exact calendar window")
    print("="*115)

    col_w = 38
    hdr   = f"{'n':>6}"
    for h in horizons:
        hdr += f"  | {h:<{col_w - 3}}"
    print(hdr)

    sub = f"{'':>6}"
    for _ in horizons:
        sub += f"  | {'Mean':>7} {'Med':>7} {'Skew':>5} {'>0':>5} {'>Tbill':>6} {'>Mkt':>6}"
    print(sub)
    print("-"*115)

    ns = results_by_horizon[horizons[0]]['n'].tolist()
    for n in ns:
        row = f"{int(n):>6}"
        for h in horizons:
            r = results_by_horizon[h]
            r = r[r['n'] == n].iloc[0]
            row += (f"  | "
                    f"{r['Mean']:>7.2%} "
                    f"{r['Median']:>7.2%} "
                    f"{r['Skew']:>5.2f} "
                    f"{r['% > 0']:>5.1%} "
                    f"{r['% > T-bill']:>6.1%} "
                    f"{r['% > VW Market']:>6.1%}")
        print(row)
    print("="*115)

def print_table12(strat_df):
    print("\n" + "="*72)
    print("TABLE 12: INDUSTRY-STRATIFIED BOOTSTRAP (full sample 1990-2024)")
    print("K = 1 to 12 (Other inkluderet)")
    print("="*72)
    print(f"{'K (industries)':>16}  {'Mean':>8}  {'Median':>8}  "
          f"{'Skew':>6}  {'%>0':>5}  {'%>Tbill':>7}  {'%>Mkt':>6}")
    print("-"*72)
    for _, row in strat_df.iterrows():
        print(f"{int(row['K']):>16}  "
              f"{row['Mean']:>8.2%}  "
              f"{row['Median']:>8.2%}  "
              f"{row['Skew']:>6.2f}  "
              f"{row['% > 0']:>5.1%}  "
              f"{row['% > T-bill']:>7.1%}  "
              f"{row['% > VW Market']:>6.1%}")
    print("="*72)

# FIGURE: DIVERSIFICATION CURVES
def plot_diversification_curves(results_by_horizon, strat_df):
    plt.rcParams.update({
        'font.family':       'serif',
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.color':        '#d4d4d4',
        'grid.linewidth':    0.6,
        'grid.alpha':        0.8,
    })

    COLORS = {
        '1-Year':      '#d1e5f0',
        '5-Year':      '#92c5de',
        '10-Year':     '#4393c3',
        'Full Sample': '#2166ac',
    }

    panel_configs = [
        ('% > T-bill',    'Probability of beating T-bills (%)',
         'Probability of Beating T-bills by Portfolio Size and Time Horizon\n'
         'Sample 1990-2024. 20,000 simulations per combination.',
         0, 100, range(0, 101, 10),
         'figure3_tbill_diversification.png'),
        ('% > VW Market', 'Probability of beating VW market (%)',
         'Probability of Beating the VW Market by Portfolio Size and Time Horizon\n'
         'Sample 1990-2024. 20,000 simulations per combination.',
         0,  40, range(0,  41,  5),
         'figure4_market_diversification.png'),
    ]

    for col, ylabel, title, ymin, ymax, yticks, fname in panel_configs:
        fig, ax = plt.subplots(figsize=(10, 5))

        for label, df_h in results_by_horizon.items():
            ax.plot(df_h['n'], df_h[col] * 100,
                    color=COLORS[label], linewidth=2,
                    marker='o', markersize=4, label=label)
        if col == '% > T-bill':
            ax.axhline(y=50, color='#6b6b6b', linewidth=1,
                   linestyle=':', alpha=0.8, label='50% threshold')

        ax.set_xlabel('Number of stocks', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set_ylim(ymin, ymax)
        ax.set_yticks(list(yticks))
        ax.set_xticks(range(0, 101, 10))
        ax.set_title(title, fontsize=11, pad=12)
        ax.legend(fontsize=8, framealpha=0.5)
        ax.tick_params(labelsize=9)

        plt.tight_layout()
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fname}")

# MAIN
if __name__ == '__main__':

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
    df = df.sort_values(['PERMNO', 'date', 'RET'], na_position='last')
    df = df.drop_duplicates(subset=['PERMNO', 'date'], keep='first')

    ff = pd.read_csv("F-F_Research_Data_Factors.CSV", skiprows=3)
    ff = ff[ff.iloc[:, 0].astype(str).str.match(r'^\s*\d{6}\s*$', na=False)].copy()
    ff.columns = ff.columns.str.strip()
    ff = ff.rename(columns={ff.columns[0]: 'date_raw'})
    ff['date_ff'] = (pd.to_datetime(ff['date_raw'].str.strip(), format='%Y%m')
                     + pd.offsets.MonthEnd(0))
    ff['RF']     = pd.to_numeric(ff['RF'],     errors='coerce') / 100
    ff['Mkt-RF'] = pd.to_numeric(ff['Mkt-RF'], errors='coerce') / 100

    df['date_month'] = df['date'] + pd.offsets.MonthEnd(0)
    df = df.merge(ff[['date_ff', 'RF']], left_on='date_month',
                  right_on='date_ff', how='left')
    df = df.drop(columns=['date_month', 'date_ff'])

    df = add_industry(df)
    entry_industry = (df.sort_values(['PERMNO', 'date'])
                        .groupby('PERMNO')['FF12']
                        .first()
                        .reset_index()
                        .rename(columns={'FF12': 'FF12_entry'}))
    df = df.drop(columns=['FF12'])
    df = df.merge(entry_industry, on='PERMNO', how='left')
    df = df.rename(columns={'FF12_entry': 'FF12'})

    print("\nComputing full-sample benchmarks...")

    tbill_monthly    = df.groupby('date')['RF'].first()
    tbill_full_gross = float((1 + tbill_monthly).prod())
    tbill_full_ret   = tbill_full_gross - 1

    ff_market = ff[['date_ff', 'Mkt-RF', 'RF']].copy()
    ff_market['Mkt'] = ff_market['Mkt-RF'] + ff_market['RF']
    sample_start = pd.Timestamp('1990-01-31')
    sample_end   = pd.Timestamp('2024-12-31')
    ff_market    = ff_market[
        (ff_market['date_ff'] >= sample_start) &
        (ff_market['date_ff'] <= sample_end)
    ].copy()

    total_months      = len(ff_market)
    market_full_gross = float((1 + ff_market['Mkt']).prod())
    market_full_ret   = market_full_gross - 1

    print(f"  Total months in sample:  {total_months}")
    print(f"  T-bill (full sample):    {tbill_full_ret:.2%}")
    print(f"  VW market (full sample): {market_full_ret:.2%}")

    annualised_market = (market_full_gross ** (1/35)) - 1
    annualised_tbill  = (tbill_full_gross  ** (1/35)) - 1
    print(f"  VW market annualised:    {annualised_market:.2%}")
    print(f"  T-bill annualised:       {annualised_tbill:.2%}")

    market_monthly = ff_market.set_index(
        ff_market['date_ff'].dt.to_period('M')
    )['Mkt'].to_dict()

    print("\nBuilding stock universe...")
    universe = build_stock_universe(df, market_monthly)
    permnos  = list(universe.keys())
    print(f"  Total stocks: {len(permnos):,}")

    # Simulation settings
    N_SIMS          = 20000
    SEED            = 1
    PORTFOLIO_SIZES = [1, 5, 10, 25, 50, 100]

    HORIZONS = {
        '1-Year':      (12,   False),
        '5-Year':      (60,   False),
        '10-Year':     (120,  False),
        'Full Sample': (None, True),
    }

    rng = np.random.default_rng(SEED)

    results_by_horizon = {}
    for label, (h_months, full) in HORIZONS.items():
        print(f"\nRunning bootstrap: {label}...")
        res = bootstrap_horizon(
            universe, permnos, PORTFOLIO_SIZES, N_SIMS, rng,
            h_months, market_full_gross, tbill_full_gross,
            full_sample=full
        )
        results_by_horizon[label] = res
        for _, row in res.iterrows():
            print(f"  n={int(row['n']):>4}  "
                  f">Tbill: {row['% > T-bill']:.1%}  "
                  f">Mkt: {row['% > VW Market']:.1%}")

    print(f"\nRunning industry-stratified bootstrap ({N_SIMS:,} sims per K)...")
    rng2     = np.random.default_rng(SEED)
    strat_df = bootstrap_stratified_fullsample(
        universe, N_SIMS, rng2, tbill_full_ret, market_full_ret
    )

    print_table11(results_by_horizon)
    print_table12(strat_df)

    plot_diversification_curves(results_by_horizon, strat_df)

    for label, df_h in results_by_horizon.items():
        fname = "q3_bootstrap_{}_90_24.csv".format(
            label.lower().replace(' ', '_').replace('-', ''))
        df_h.to_csv(fname, index=False)
    strat_df.to_csv("q3_bootstrap_stratified_90_24.csv", index=False)

    print("\nAll outputs saved.")