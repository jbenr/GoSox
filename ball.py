import pandas as pd
import pybaseball as pyb
import utils
import os
from datetime import datetime, date
import data_pullson
import data_crunchski
import modelo

# https://baseballsavant.mlb.com/csv-docs

# player_ids = ['815084','814217','814005']
# data = pyb.playerid_reverse_lookup(key_type='mlbam')
# print(data)

def run():
    df = data_crunchski.prep_test_train()
    pred = modelo.modelo(df)


if __name__ == "__main__":
    # data_pullson.pull_sched(start_year=date.today().year, end_year=date.today().year)
    # data_pullson.pull_the_stat_dat(start_year=date.today().year, end_year=date.today().year)

    df = pd.read_parquet("data/statcast/statcast_2025.parquet")
    print(df.description.unique())

    nan_percent = df.isna().mean().mul(100).round(2).reset_index()
    nan_percent.columns = ['column', 'percent_nan']
    # utils.pdf(nan_percent, format=8)

    for col in df.columns:
        if 'angle' in col:
            print(col)

    df['strike'] = df['description'].isin([
        'swinging_strike', 'foul', 'called_strike', 'foul_tip', 'swinging_strike_blocked'
    ]).astype(int)
    df['ball'] = (df['description'] == 'ball').astype(int)
    utils.pdf(df.dropna(subset='arm_angle').tail(5), format=8)

    dialed = df.groupby('player_name').agg({'ball':'sum','strike':'sum','arm_angle':'mean'}).reset_index()
    dialed['pitch_count'] = dialed['ball']+dialed['strike']
    dialed['ratio'] = dialed['strike']/dialed['ball']
    dialed = dialed[(dialed['ball']>0) & (dialed['pitch_count']>1000)]
    utils.pdf(dialed.sort_values(by='arm_angle',ascending=False).tail(10))
