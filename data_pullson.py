import pandas as pd
import pybaseball as pyb
import utils
from datetime import datetime
from tqdm import tqdm
import warnings
import requests
import os

warnings.filterwarnings("ignore", category=FutureWarning)


def pull_the_stat_dat(start_year=2021, end_year=datetime.today().year):
    utils.make_dir('data/statcast')

    for year in tqdm(range(start_year, end_year + 1), desc="Pulling Statcast data"):
        df = pyb.statcast(f'{year}-03-01', f'{year}-12-01', parallel=True)
        df.to_parquet(f'data/statcast/statcast_{year}.parquet')

        # utils.make_dir(f'data/{year}')
        # a = f"{year}-01-01"
        # b = f"{year}-01-02"
        # date_range = pd.date_range(start=a, end=b, freq='D')
        # formatted_dates = date_range.strftime('%Y-%m-%d').tolist()
        #
        # for d in formatted_dates:
        #     df = pyb.statcast(d,d)
        #     if len(df) > 0: df.to_parquet(f'data/statcast_{d}.parquet')


def pull_sched(start_year=2021, end_year=datetime.today().year):
    utils.make_dir('data/sched')

    for year in tqdm(range(start_year, end_year + 1), desc="Pulling schedules"):
        url = f"https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/{year}/{year}schedule.csv"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            outfile = os.path.join('data/sched', f"{year}schedule.csv")
            with open(outfile, "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to retrieve {year} schedule. HTTP {response.status_code} - {response.reason}")


if __name__ == "__main__":
    pull_sched(start_year=2025)
    pull_the_stat_dat(start_year=2025)