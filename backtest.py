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


if __name__ == "__main__":
    # odds = oddson()
    # utils.pdf(odds.tail(20))
    df = concat_parquet_files('data/bt/backtest_2022_2025_150/preds')
    # utils.pdf(df)
    df.sort_values(by='game_date',inplace=True)
    # utils.pdf(df.tail(20))

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

    # df = df[df['var'] <= 0.2]
    df = df[df['plus_odds_bet'] == True]
    # df = df[df['abs_diff'] >= 0.6]
    df = df[df['conf'] >= 5]
    utils.pdf(df.tail(20))

    g = df.groupby('mispricing_bucket').agg({'win?':'sum','pred':'count'}).reset_index()
    # g = df.groupby('conf_bucket').agg({'win?':'sum','pred':'count'}).reset_index()

    g['%'] = g['win?'] / g['pred']
    utils.pdf(g)

    print(str(100*(df['win?'].sum()/len(df)).round(4)), '%', f" {len(df)}")
