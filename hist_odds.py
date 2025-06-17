import os
import time
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import utils  # Assuming you have a utils module with oh_waiter or similar

def scrape_strikeout_odds_selenium(date_str="2024-03-28"):
    url = f"https://www.bettingpros.com/mlb/odds/player-props/strikeouts/?date={date_str}"
    opts = Options()
    opts.add_argument("--headless")
    driver = webdriver.Chrome(options=opts)
    driver.get(url)

    utils.oh_waiter(3, "initial load")
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    data_list = []
    offers = soup.select("div.flex.odds-offer")
    for offer in offers:
        pitcher_el = offer.select_one("a.odds-player__heading")
        pitcher_name = pitcher_el.get_text(strip=True) if pitcher_el else "Unknown"

        match_el = offer.select_one("div.odds-player__matchup-tag")
        matchup_str = match_el.get_text(strip=True) if match_el else "Unknown at Unknown"
        away_team, home_team = "Unknown", "Unknown"
        if " at " in matchup_str:
            away_team, home_team = matchup_str.split(" at ", 1)

        col_elems = offer.select("div.odds-offer__item")
        consensus_ou = None
        over_odds = None
        under_odds = None

        if col_elems:
            consensus_col = col_elems[-1]
            btns = consensus_col.select("button.odds-cell")
            for b in btns:
                line_el = b.select_one("span.odds-cell__line")
                cost_el = b.select_one("span.odds-cell__cost")
                if line_el and cost_el:
                    line_txt = line_el.get_text(strip=True)
                    cost_txt = cost_el.get_text(strip=True).strip("()")
                    if " " in line_txt:
                        direction, ou_number = line_txt.split(" ", 1)
                        if direction.upper() == "O":
                            consensus_ou = ou_number
                            over_odds = cost_txt
                        elif direction.upper() == "U":
                            under_odds = cost_txt

        data_list.append({
            "pitcher": pitcher_name,
            "away_team": away_team,
            "home_team": home_team,
            "consensus_ou": consensus_ou,
            "consensus_over_odds": over_odds,
            "consensus_under_odds": under_odds,
        })

    return pd.DataFrame(data_list)

if __name__ == "__main__":
    base_path = "data/strikeout_odds"
    utils.make_dir(base_path)
    years = [2024]  # or any list of years

    for year in years:
        utils.make_dir(base_path+f'/{year}')
        start_date = datetime.date(year, 3, 1)
        end_date = datetime.date(year, 11, 1)
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Scraping {date_str} ...")
            df = scrape_strikeout_odds_selenium(date_str)

            if len(df)>0:
                fn_date = current_date.strftime("%m_%d_%Y")
                out_file = f'{base_path}/{year}/strikout_odds_{fn_date}.parquet'
                df.to_parquet(out_file)
                print(f"Saved: {out_file}")

            current_date += datetime.timedelta(days=1)
