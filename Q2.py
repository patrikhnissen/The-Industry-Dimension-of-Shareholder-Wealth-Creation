import pandas as pd
import numpy as np


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
    df = df.drop(columns=['FF12_timevary'])
    df = df.merge(entry_industry, on='PERMNO', how='left')
    return df



# WEALTH CREATION PER STOCK (FULL SAMPLE)
def compute_wealth_creation(df):
    df = df.sort_values(['PERMNO', 'date'])
    records = []
    for permno, group in df.groupby('PERMNO'):
        gross_stock  = (1 + group['RET']).prod()
        gross_tbill  = (1 + group['RF']).prod()
        bh_return    = gross_stock - 1
        tbill_return = gross_tbill - 1
        me_0         = group['MCAP'].iloc[0]
        wc           = me_0 * (gross_stock - gross_tbill)
        industry     = group['FF12'].iloc[0] if 'FF12' in group.columns \
                       else 'Unknown'
        records.append({
            'PERMNO':          permno,
            'FF12':            industry,
            'n_months':        len(group),
            'bh_return':       bh_return,
            'tbill_return':    tbill_return,
            'beat_tbill':      int(bh_return > tbill_return),
            'positive':        int(bh_return > 0),
            'me_0':            me_0,
            'wealth_creation': wc,
            'positive_wc':     int(wc > 0),
        })
    return pd.DataFrame(records)



# FF12 ORDER
FF12_ORDER = [
    'Consumer Non-Durables', 'Consumer Durables', 'Manufacturing',
    'Energy', 'Chemicals', 'Business Equipment', 'Telecoms',
    'Utilities', 'Shops', 'Healthcare', 'Finance', 'Other'
]

SUB_PERIODS = {
    'P1: 1990-1999': ('1990-01-01', '1999-12-31'),
    'P2: 2000-2009': ('2000-01-01', '2009-12-31'),
    'P3: 2010-2024': ('2010-01-01', '2024-12-31'),
}



# SHAREHOLDER WEALTH CREATION BY FAMA-FRENCH 12 INDUSTRY
def table6_full_sample(stock_df, df_monthly):
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
        n_fm     = len(sub_fm)
        wc_share = ind_wc / total_wc if total_wc != 0 else np.nan
        fm_share = n_fm / total_fm if total_fm != 0 else np.nan
        ratio    = wc_share / fm_share if fm_share != 0 else np.nan
        rows.append({
            'Industry':    ind,
            'WC ($m)':     ind_wc / 1000,
            'Firm/Months': n_fm,
            'WC%':         wc_share,
            'FM%':         fm_share,
            'Ratio WC/FM': ratio,
        })

    summary = pd.DataFrame(rows)

    print("\n" + "="*100)
    print("TABLE 6: SHAREHOLDER WEALTH CREATION BY INDUSTRY — FULL SAMPLE 1990-2024")
    print("Following Bessembinder (2020) Exhibit 2 structure")
    print("="*100)
    print(f"{'Industry':<26} {'WC ($m)':>14}  {'Firm/Months':>12}  "
          f"{'WC%':>7}  {'FM%':>7}  {'Ratio':>7}")
    print("-"*100)
    for _, row in summary.iterrows():
        print(f"{row['Industry']:<26} "
              f"{row['WC ($m)']:>14,.0f}  "
              f"{int(row['Firm/Months']):>12,}  "
              f"{row['WC%']:>7.2%}  "
              f"{row['FM%']:>7.2%}  "
              f"{row['Ratio WC/FM']:>7.2f}")
    print("-"*100)
    print(f"{'TOTAL':<26} "
          f"{total_wc_m:>14,.0f}  "
          f"{total_fm:>12,}  "
          f"{'100.00%':>7}  "
          f"{'100.00%':>7}  "
          f"{'1.00':>7}")
    print("="*100)

    summary.to_csv("table_full_sample_90_24.csv", index=False)
    print("Saved: table_full_sample_90_24.csv")
    return summary



# SUB-PERIOD WEALTH CREATION
def compute_subperiod_wealth(df, start, end):
    mask = (df['date'] >= start) & (df['date'] <= end)
    sub  = df[mask].copy().sort_values(['PERMNO', 'date'])

    records = []
    for permno, group in sub.groupby('PERMNO'):
        gross_stock = (1 + group['RET']).prod()
        gross_tbill = (1 + group['RF']).prod()
        bh_return   = gross_stock - 1
        me_0        = group['MCAP'].iloc[0]
        wc          = me_0 * (gross_stock - gross_tbill)
        industry    = group['FF12'].iloc[0] if 'FF12' in group.columns \
                      else 'Unknown'

        if 'COMNAM' in group.columns:
            name = group['COMNAM'].dropna().iloc[-1] \
                   if not group['COMNAM'].dropna().empty else 'Unknown'
        else:
            name = 'Unknown'

        if 'TICKER' in group.columns:
            ticker = group['TICKER'].dropna().iloc[-1] \
                     if not group['TICKER'].dropna().empty else ''
        else:
            ticker = ''

        records.append({
            'PERMNO':          permno,
            'Name':            name,
            'Ticker':          ticker,
            'FF12':            industry,
            'n_months':        len(group),
            'bh_return':       bh_return,
            'me_0':            me_0,
            'wealth_creation': wc,
        })
    return pd.DataFrame(records)


def table6_subperiod(stock_sp, df_sp, period_label):
    total_wc   = stock_sp['wealth_creation'].sum()
    total_wc_m = total_wc / 1000
    total_fm   = len(df_sp)

    rows = []
    for ind in FF12_ORDER:
        sub    = stock_sp[stock_sp['FF12'] == ind]
        sub_fm = df_sp[df_sp['FF12'] == ind]
        if len(sub) == 0:
            continue
        ind_wc   = sub['wealth_creation'].sum()
        n_fm     = len(sub_fm)
        wc_share = ind_wc / total_wc if total_wc != 0 else np.nan
        fm_share = n_fm / total_fm if total_fm != 0 else np.nan
        ratio    = wc_share / fm_share if fm_share != 0 else np.nan
        rows.append({
            'Industry':    ind,
            'WC ($m)':     ind_wc / 1000,
            'Firm/Months': n_fm,
            'WC%':         wc_share,
            'FM%':         fm_share,
            'Ratio WC/FM': ratio,
            'Period':      period_label,
        })

    summary = pd.DataFrame(rows)

    pos_wc_industries = summary[summary['WC ($m)'] > 0].copy()
    gross_wc_m = pos_wc_industries['WC ($m)'].sum()

    if gross_wc_m > 0:
        pos_wc_industries = pos_wc_industries.copy()
        pos_wc_industries['gross_share'] = (
            pos_wc_industries['WC ($m)'] / gross_wc_m
        )
        hhi = float((pos_wc_industries['gross_share'] ** 2).sum())
        cr3 = float(pos_wc_industries.nlargest(3, 'WC ($m)')['gross_share'].sum())
    else:
        hhi = np.nan
        cr3 = np.nan

    hhi_note     = " (gross WC basis)"
    wc_sign_note = ""
    if total_wc < 0:
        wc_sign_note = "  *** AGGREGATE WC IS NEGATIVE THIS PERIOD ***"
        print(f"  [INFO] Negative total WC detected in {period_label}.")
        print(f"  [INFO] Gross WC = ${gross_wc_m:,.0f}m from "
              f"{len(pos_wc_industries)} positive industries.")

    print("\n" + "="*100)
    print(f"TABLE 6: {period_label}  |  Total WC: ${total_wc_m:,.0f}m  |  "
          f"HHI: {hhi:.4f}{hhi_note}  |  CR3: {cr3:.2%}{hhi_note}"
          f"{wc_sign_note}")
    print("="*100)
    print(f"{'Industry':<26} {'WC ($m)':>14}  {'Firm/Months':>12}  "
          f"{'WC%':>7}  {'FM%':>7}  {'Ratio':>7}")
    print("-"*100)
    for _, row in summary.iterrows():
        print(f"{row['Industry']:<26} "
              f"{row['WC ($m)']:>14,.0f}  "
              f"{int(row['Firm/Months']):>12,}  "
              f"{row['WC%']:>7.2%}  "
              f"{row['FM%']:>7.2%}  "
              f"{row['Ratio WC/FM']:>7.2f}")
    print("-"*100)
    print(f"{'TOTAL':<26} "
          f"{total_wc_m:>14,.0f}  "
          f"{total_fm:>12,}  "
          f"{'100.00%':>7}  "
          f"{'100.00%':>7}  "
          f"{'1.00':>7}")
    if total_wc < 0:
        print("NOTE: WC% and Ratio use negative total WC as denominator and are")
        print("      not directly interpretable. HHI and CR3 use gross WC basis.")
        print(f"      Gross WC (positive industries only): ${gross_wc_m:,.0f}m")
    print("="*100)

    summary['HHI']           = hhi
    summary['CR3']           = cr3
    summary['HHI_note']      = hhi_note.strip()
    summary['Gross WC ($m)'] = gross_wc_m
    return summary


def run_q2_analysis(df):
    all_summaries = []
    hhi_rows      = []

    for label, (start, end) in SUB_PERIODS.items():
        print(f"\nComputing sub-period: {label}...")
        stock_sp = compute_subperiod_wealth(df, start, end)
        mask_sp  = (df['date'] >= start) & (df['date'] <= end)
        df_sp    = df[mask_sp].copy()

        sp_summary = table6_subperiod(stock_sp, df_sp, label)
        all_summaries.append(sp_summary)

        hhi_rows.append({
            'Period':        label,
            'N firms':       len(stock_sp),
            'Total WC ($m)': stock_sp['wealth_creation'].sum() / 1000,
            'Gross WC ($m)': sp_summary['Gross WC ($m)'].iloc[0],
            'HHI':           sp_summary['HHI'].iloc[0],
            'CR3':           sp_summary['CR3'].iloc[0],
        })

    hhi_df = pd.DataFrame(hhi_rows)
    print("\n" + "="*65)
    print("Q2: HHI AND CR3 ACROSS SUB-PERIODS")
    print("="*65)
    print(f"{'Period':<20} {'N Firms':>8}  {'WC ($m)':>14}  "
          f"{'HHI':>8}  {'CR3':>8}")
    print("-"*65)
    for _, row in hhi_df.iterrows():
        print(f"{row['Period']:<20} {int(row['N firms']):>8}  "
              f"{row['Total WC ($m)']:>14,.0f}  "
              f"{row['HHI']:>8.4f}  "
              f"{row['CR3']:>8.2%}")
    print("="*65)
    print("NOTE: HHI and CR3 computed on gross WC basis (positive industries only)")
    print("      in all periods for consistency. P2 has negative total WC.")

    combined = pd.concat(all_summaries, ignore_index=True)
    return combined, hhi_df



# HHI AND CR3 STANDALONE
def table9_hhi_cr3(hhi_df):
    print("\n" + "="*90)
    print("HHI AND CR3 CONCENTRATION ACROSS SUB-PERIODS")
    print("="*90)
    print(f"{'Period':<20} {'N Firms':>8}  {'Total WC ($m)':>14}  "
          f"{'Gross WC ($m)':>14}  {'HHI':>8}  {'CR3':>8}")
    print("-"*90)
    for _, row in hhi_df.iterrows():
        wc_note = '*' if row['Total WC ($m)'] < 0 else ' '
        print(f"{row['Period']:<20} {int(row['N firms']):>8}  "
              f"{row['Total WC ($m)']:>14,.0f}{wc_note} "
              f"{row['Gross WC ($m)']:>14,.0f}  "
              f"{row['HHI']:>8.4f}  "
              f"{row['CR3']:>8.2%}")
    print("="*90)
    print("Gross WC = sum of positive-WC industries only. HHI and CR3 are")
    print("computed relative to Gross WC in all periods for consistency.")
    print("* Total WC is negative in this period due to aggregate wealth destruction.")

    hhi_df.to_csv("table_hhi_cr3_90_24.csv", index=False)
    print("Saved: table_hhi_cr3_90_24.csv")

# TOP 5 INDUSTRIES BY SUB-PERIOD
def table10_top5_by_period(q2_combined):
    periods = q2_combined['Period'].unique()

    top5_by_period = {}
    for period in periods:
        sub = q2_combined[q2_combined['Period'] == period].copy()
        sub = sub[sub['WC ($m)'] > 0]
        sub = sub.sort_values('WC ($m)', ascending=False).head(5)
        top5_by_period[period] = sub[['Industry', 'WC ($m)',
                                      'FM%']].reset_index(drop=True)

    print("\n" + "="*105)
    print("TOP 5 WEALTH-CREATING INDUSTRIES BY SUB-PERIOD")
    print("Only industries with positive WC included. Ranked by WC ($m).")
    print("="*105)

    hdr = f"{'Rank':>5}"
    for period in periods:
        hdr += f"  | {period:<32}"
    print(hdr)

    sub_hdr = f"{'':>5}"
    for _ in periods:
        sub_hdr += f"  | {'Industry':<22} {'WC ($m)':>10} {'FM%':>6}"
    print(sub_hdr)
    print("-"*105)

    for rank in range(5):
        row_str = f"{rank+1:>5}"
        for period in periods:
            df_p = top5_by_period[period]
            if rank < len(df_p):
                r = df_p.iloc[rank]
                row_str += (f"  | {str(r['Industry']):<22} "
                            f"{r['WC ($m)']:>10,.0f} "
                            f"{r['FM%']:>6.2%}")
            else:
                row_str += f"  | {'—':<22} {'':>10} {'':>6}"
        print(row_str)

    print("="*105)
    print("WC ($m) = absolute wealth creation in USD millions within the period.")
    print("FM%     = industry share of total firm/months in the period.")
    print("Note: P2 (2000-2009) shows only industries that created positive")
    print("      wealth despite aggregate market wealth destruction in that decade.")

    rows = []
    for rank in range(5):
        row = {'Rank': rank + 1}
        for period in periods:
            df_p = top5_by_period[period]
            if rank < len(df_p):
                r = df_p.iloc[rank]
                row[f'{period} Industry'] = r['Industry']
                row[f'{period} WC ($m)']  = r['WC ($m)']
                row[f'{period} FM%']      = r['FM%']
        rows.append(row)
    pd.DataFrame(rows).to_csv("table10_top5_by_period_90_24.csv", index=False)
    print("Saved: table10_top5_by_period_90_24.csv")

# TOP 5 WEALTH CREATORS AND DESTROYERS BY SUB-PERIOD
def table_top5_creators_destroyers(df):
    print("\n" + "="*110)
    print("TOP 5 WEALTH CREATORS AND DESTROYERS BY SUB-PERIOD")
    print("="*110)

    all_rows = []

    for label, (start, end) in SUB_PERIODS.items():
        period_df = compute_subperiod_wealth(df, start, end)
        total_wc  = period_df['wealth_creation'].sum()

        creators   = period_df.nlargest(5,  'wealth_creation')
        destroyers = period_df.nsmallest(5, 'wealth_creation')

        print(f"\n{'─'*110}")
        print(f"  {label}  |  Total WC: ${total_wc/1e3:,.0f}m  "
              f"|  N firms: {len(period_df):,}")
        print(f"{'─'*110}")

        print(f"\n  TOP 5 WEALTH CREATORS:")
        print(f"  {'Rank':<5} {'Name':<35} {'Ticker':<8} {'Industry':<26} "
              f"{'WC ($m)':>12}  {'BHR':>8}  {'WC share':>9}")
        print(f"  {'-'*103}")
        for rank, (_, row) in enumerate(creators.iterrows(), 1):
            wc_share = row['wealth_creation'] / total_wc \
                       if total_wc != 0 else np.nan
            print(f"  {rank:<5} {str(row['Name'])[:34]:<35} "
                  f"{str(row['Ticker'])[:7]:<8} "
                  f"{str(row['FF12'])[:25]:<26} "
                  f"{row['wealth_creation']/1e3:>12,.0f}  "
                  f"{row['bh_return']:>8.1%}  "
                  f"{wc_share:>9.2%}")

        print(f"\n  TOP 5 WEALTH DESTROYERS:")
        print(f"  {'Rank':<5} {'Name':<35} {'Ticker':<8} {'Industry':<26} "
              f"{'WC ($m)':>12}  {'BHR':>8}  {'WC share':>9}")
        print(f"  {'-'*103}")
        for rank, (_, row) in enumerate(destroyers.iterrows(), 1):
            wc_share = row['wealth_creation'] / total_wc \
                       if total_wc != 0 else np.nan
            print(f"  {rank:<5} {str(row['Name'])[:34]:<35} "
                  f"{str(row['Ticker'])[:7]:<8} "
                  f"{str(row['FF12'])[:25]:<26} "
                  f"{row['wealth_creation']/1e3:>12,.0f}  "
                  f"{row['bh_return']:>8.1%}  "
                  f"{wc_share:>9.2%}")

        for rank, (_, row) in enumerate(creators.iterrows(), 1):
            all_rows.append({
                'Period':            label,
                'Type':              'Creator',
                'Rank':              rank,
                'PERMNO':            row['PERMNO'],
                'Name':              row['Name'],
                'Ticker':            row['Ticker'],
                'Industry':          row['FF12'],
                'WC ($m)':           row['wealth_creation'] / 1e3,
                'BH Return':         row['bh_return'],
                'WC share of total': row['wealth_creation'] / total_wc
                                     if total_wc != 0 else np.nan,
            })
        for rank, (_, row) in enumerate(destroyers.iterrows(), 1):
            all_rows.append({
                'Period':            label,
                'Type':              'Destroyer',
                'Rank':              rank,
                'PERMNO':            row['PERMNO'],
                'Name':              row['Name'],
                'Ticker':            row['Ticker'],
                'Industry':          row['FF12'],
                'WC ($m)':           row['wealth_creation'] / 1e3,
                'BH Return':         row['bh_return'],
                'WC share of total': row['wealth_creation'] / total_wc
                                     if total_wc != 0 else np.nan,
            })

    print(f"\n{'═'*110}")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv("table_top5_creators_destroyers_by_period.csv", index=False)
    print("Saved: table_top5_creators_destroyers_by_period.csv")
    return result_df

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
    df = df.merge(ff[['date_ff', 'RF']], left_on='date_month', right_on='date_ff',
                  how='left')
    df = df.drop(columns=['date_month', 'date_ff'])

    n_missing_rf = df['RF'].isna().sum()
    if n_missing_rf > 0:
        print(f"WARNING: {n_missing_rf} rows have no RF match.")
    else:
        print("RF merged successfully.")

    df = add_industry(df)
    df = apply_entry_sic(df)

    print("\nComputing full-sample per-stock wealth creation...")
    stock_df = compute_wealth_creation(df)
    table6_full_sample(stock_df, df)

    print("\n" + "="*65)
    print("Q2: TABLE 6 BY SUB-PERIOD")
    print("="*65)
    q2_combined, q2_hhi = run_q2_analysis(df)

    table9_hhi_cr3(q2_hhi)

    table10_top5_by_period(q2_combined)

    print("\nComputing top 5 wealth creators and destroyers by sub-period...")
    table_top5_creators_destroyers(df)

    q2_hhi.to_csv("q2_hhi_cr3_90_24.csv", index=False)

    col_map = {
        'Industry':    'Industry',
        'WC ($m)':     'WC ($m)',
        'Firm/Months': 'Firm/Months',
        'WC%':         'WC%',
        'FM%':         'Firm/Months%',
        'Ratio WC/FM': 'Ratio WC/FM',
    }
    for period in SUB_PERIODS.keys():
        sub = q2_combined[q2_combined['Period'] == period].copy()
        sub = sub[list(col_map.keys())].rename(columns=col_map)
        total_row = pd.DataFrame([{
            'Industry':      'Total',
            'WC ($m)':       sub['WC ($m)'].sum(),
            'Firm/Months':   sub['Firm/Months'].sum(),
            'WC%':           sub['WC%'].sum(),
            'Firm/Months%':  sub['Firm/Months%'].sum(),
            'Ratio WC/FM':   np.nan,
        }])
        sub = pd.concat([sub, total_row], ignore_index=True)
        period_clean = period.replace(': ', '_').replace('-', '_').replace(' ', '_')
        fname = f"q2_table6_{period_clean}.csv"
        sub.to_csv(fname, index=False)
        print(f"Saved: {fname}")

    print("\nAll outputs saved:")
    print("  table6_full_sample_90_24.csv")
    print("  q2_table6_P1_1990_1999.csv")
    print("  q2_table6_P2_2000_2009.csv")
    print("  q2_table6_P3_2010_2024.csv")
    print("  table9_hhi_cr3_90_24.csv")
    print("  table10_top5_by_period_90_24.csv")
    print("  table_top5_creators_destroyers_by_period.csv")