import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import utils


def oddson(start_year=2021, end_year=2024, odds_dir="data/strikeout_odds"):
    ############################################################################
    # Read odds
    ############################################################################
    if not os.path.isdir(odds_dir):
        print("No odds directory found for odds.")
        return

    all_odds = []
    for yr in range(start_year, end_year + 1):
        for item in sorted(os.listdir(odds_dir+'/'+str(yr))):
            if item.endswith(".parquet"):
                base = item.replace(".parquet", "")
                # Expect: strikout_odds_MM_DD_YYYY
                # or whatever your naming scheme is:
                # e.g. "strikout_odds_07_31_2024.parquet"
                parts = base.split("_")
                # last 3 parts: mm, dd, yyyy
                mm, dd, yyyy = parts[-3], parts[-2], parts[-1]
                d = date(int(yyyy), int(mm), int(dd))
                df_pq = pd.read_parquet(os.path.join(odds_dir+'/'+str(yr), item))
                df_pq["game_date"] = d
                all_odds.append(df_pq)
    if all_odds:
        odds = pd.concat(all_odds, ignore_index=True).sort_values("game_date")
        odds["consensus_ou"] = pd.to_numeric(odds["consensus_ou"], errors="coerce")
    else:
        odds = pd.DataFrame()

    # # Merge with pitcher aggregated
    # final_pitchers = pd.merge(
    #     roll_df_pitchers,
    #     odds,
    #     on=["game_date", "pitcher", "away_team", "home_team"],
    #     how="left"
    # )
    # odds["result"] = np.where(
    #     final_pitchers["is_k"] > final_pitchers["consensus_ou"], "over",
    #     np.where(final_pitchers["is_k"] < final_pitchers["consensus_ou"], "under", "push")
    # )

    return odds


def concat_parquet_files(path):
    df_list = []

    for file in os.listdir(path):
        if file.endswith(".parquet"):
            file_path = os.path.join(path, file)
            df = pd.read_parquet(file_path)
            df_list.append(df)

    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        return final_df
    else:
        return pd.DataFrame()


def american_to_prob(odds):
    if isinstance(odds, str):
        odds = odds.strip().upper()
        if odds == "EVEN":
            return 0.5
        odds = int(odds)

    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def american_odds_to_multiplier(odds):
    if isinstance(odds, str):
        odds = odds.strip().upper()
        if odds == "EVEN":
            return 2.0  # Even odds = 1:1 payout
        odds = int(odds)
    return 1 + (odds / 100) if odds > 0 else 1 + (100 / -odds)


if __name__ == "__main__":
    # odds = oddson()
    # utils.pdf(odds.tail(20))
    df = concat_parquet_files('data/bt/backtest_2022_2025_150/preds')
    df['game_date'] = pd.to_datetime(df['game_date'])
    # utils.pdf(df)
    df.sort_values(by='game_date',inplace=True)
    # utils.pdf(df.tail(20))

    df['prob_over'] = df['consensus_over_odds'].apply(american_to_prob)
    df['prob_under'] = df['consensus_under_odds'].apply(american_to_prob)
    df['prob_selected'] = df.apply(
        lambda row: row['prob_over'] if row['bet'].lower() == 'over' else row['prob_under'],
        axis=1
    )

    df['odds_selected'] = df.apply(
        lambda row: row['consensus_over_odds'] if row['bet'].lower() == 'over' else row['consensus_under_odds'],
        axis=1
    )

    df['payout_multiplier'] = df['odds_selected'].apply(american_odds_to_multiplier)

    stake = 100
    df['profit'] = df.apply(
        lambda row: (row['payout_multiplier'] * stake - stake) if row['win?'] == 1 else -stake,
        axis=1
    )

    # df = df[df['var']<0.1]
    max_cutoff = 1
    bucket_size = 0.1
    bins = np.arange(0, max_cutoff + bucket_size, bucket_size)
    df['var_bucket'] = pd.cut(df['var'], bins=bins, right=False)
    df['var_bucket'] = df['var_bucket'].astype(str)

    df['conf'] = df['abs_diff'] / np.sqrt(df['var'])
    max_cutoff = 15
    bucket_size = 1
    bins = np.arange(0, max_cutoff + bucket_size, bucket_size)
    df['conf_bucket'] = pd.cut(df['conf'], bins=bins, right=False)
    df['conf_bucket'] = df['conf_bucket'].astype(str)

    df = df[df['var'] <= 0.2]
    df = df[df['plus_odds_bet'] == True]
    df = df[df['abs_diff'] >= 1.6]
    # df = df[df['conf'] >= 5]
    utils.pdf(df.tail(8))

    g = df.groupby('mispricing_bucket').agg({'win?':'sum','pred':'count'}).reset_index()
    # g = df.groupby('conf_bucket').agg({'win?':'sum','pred':'count'}).reset_index()

    g['%'] = g['win?'] / g['pred']
    utils.pdf(g)

    print(str(100*(df['win?'].sum()/len(df)).round(4)), '%', f" {len(df)}")

    df['year'] = df['game_date'].dt.year
    d = df.groupby('year').agg({'win?':'sum','result':'count', 'profit':'mean'}).reset_index()
    d_ = df.groupby('year').agg({'profit':'sum'}).reset_index()
    d_.rename(columns={'profit':'total'}, inplace=True)
    d = pd.merge(d, d_, on='year')
    d['%'] = d['win?'] / d['result'] * 100
    d['roi'] = d['total'] / d['result']
    utils.pdf(d)

    average_probability = df['prob_selected'].mean()
    print(f"Average probability of bets: {average_probability:.4f}")

    avg_payout = df.loc[df['win?'] == 1, 'payout_multiplier'].mean() * stake
    total_profit = df['profit'].sum()
    avg_profit_per_bet = df['profit'].mean()
    roi = (total_profit / (len(df) * stake)) * 100

    print(f"Number of bets: {len(df)}")
    print(f"Average payout per winning bet: ${avg_payout:.2f}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"ROI: {roi:.2f}%")

    thresholds = np.arange(0, 3.1, 0.05)  # e.g., 0.0, 0.1, 0.2, ... 3.0
    cumulative_stats = []

    for t in thresholds:
        subset = df[df['var'] >= t]
        if len(subset) == 0:
            continue
        win_count = subset['win?'].sum()
        bet_count = len(subset)
        win_rate = win_count / bet_count
        total_profit = subset['profit'].sum()
        roi = total_profit / (bet_count * stake)
        cumulative_stats.append({
            'threshold': round(t, 2),
            'bets': bet_count,
            'wins': win_count,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi
        })

    g_cumulative = pd.DataFrame(cumulative_stats)
    utils.pdf(g_cumulative)