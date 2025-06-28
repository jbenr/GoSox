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
    df.tail(3)
    # pred = run()