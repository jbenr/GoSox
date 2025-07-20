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



def live_mlb_pitcher_k_props():
    API_KEY = '7f0d888986edaf32491f95580e31a0dd'
    sport = 'baseball_mlb'

    url_events = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
    params_events = {
        'api_key': API_KEY,
        'regions': 'us',
        'markets': 'totals',
        'oddsFormat': 'decimal',
        'dateFormat': 'iso',
    }

    r_events = requests.get(url_events, params=params_events)
    events = r_events.json()
    print(f"Fetched {len(events)} events")

    all_props = []

    for event in events:
        event_id = event['id']
        home = event.get("home_team")
        away = event.get("away_team")
        commence = event.get("commence_time")

        print(f"\nChecking event: {away} @ {home}")

        url_props = f'https://api.the-odds-api.com/v4/sports/{sport}/events/{event_id}/odds'
        params_props = {
            'api_key': API_KEY,
            'regions': 'us',
            'markets': 'pitcher_strikeouts',
            'oddsFormat': 'decimal',
            'dateFormat': 'iso',
        }

        r_props = requests.get(url_props, params=params_props)
        if r_props.status_code != 200:
            print(f"  ❌ Error {r_props.status_code}: {r_props.text}")
            continue

        data = r_props.json()
        if not data.get("bookmakers"):
            print("  ❌ No pitcher props available.")
            continue

        print("  ✅ Pitcher props found!")

        for bookmaker in data.get("bookmakers", []):
            book_name = bookmaker["title"]
            for market in bookmaker.get("markets", []):
                if market["key"] != "pitcher_strikeouts":
                    continue
                for outcome in market.get("outcomes", []):
                    all_props.append({
                        "event_id": event_id,
                        "date": commence,
                        "away_team": away,
                        "home_team": home,
                        "bookmaker": book_name,
                        "player": outcome.get("name"),
                        "line": outcome.get("point"),
                        "odds": outcome.get("price")
                    })

    props_df = pd.DataFrame(all_props)
    utils.pdf(props_df)

    print("\nRequests remaining:", r_props.headers.get("x-requests-remaining"))
    print("Requests used:", r_props.headers.get("x-requests-used"))

    return props_df


if __name__ == "__main__":
    # pull_sched(start_year=2025)
    # pull_the_stat_dat(start_year=2025)

    live_mlb_pitcher_k_props()