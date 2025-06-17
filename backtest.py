import pandas as pd
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


if __name__ == "__main__":
    odds = oddson()
    utils.pdf(odds.tail(20))