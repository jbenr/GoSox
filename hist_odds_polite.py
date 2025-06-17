import os
import time
import random
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import utils  # Assuming you have a utils module with oh_waiter or similar

def scrape_strikeout_odds_selenium(date_str="2024-03-28"):
    # Polite approach: use a realistic User-Agent
    # Example: an up-to-date Chrome UA on Windows
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/116.0.0.0 Safari/537.36"
    )

    url = f"https://www.bettingpros.com/mlb/odds/player-props/strikeouts/?date={date_str}"

    chrome_opts = Options()
    # Headless is helpful but sometimes more easily detected
    # You can comment this out if you want a headful browser
    chrome_opts.add_argument("--headless")

    # Apply our custom user-agent
    chrome_opts.add_argument(f"--user-agent={user_agent}")

    driver = webdriver.Chrome(options=chrome_opts)
    driver.get(url)

    # Random small delay after initial load
    time.sleep(random.uniform(5.0, 10.0))

    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Again, randomize the scrolling wait
        time.sleep(random.uniform(4.0, 6.0))
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
            # The final column is typically the consensus column
            consensus_col = col_elems[-1]
            btns = consensus_col.select("button.odds-cell")
            for b in btns:
                line_el = b.select_one("span.odds-cell__line")
                cost_el = b.select_one("span.odds-cell__cost")
                if line_el and cost_el:
                    line_txt = line_el.get_text(strip=True)
                    # Remove parentheses from the cost
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
    # years = [2021,2022,2023]  # or any list of years
    # years = list(range(2010,2023+1))
    years = [2025]

    for year in years:
        # Make directory for that year
        year_path = os.path.join(base_path, str(year))
        utils.make_dir(year_path)

        start_date = datetime.date(year, 3, 1)
        end_date = datetime.date(year, 11, 5)
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            print(f"Scraping {date_str} ...")

            # Add a random delay between days to avoid suspicious repeated requests
            time.sleep(random.uniform(1.0, 3.0))

            df = scrape_strikeout_odds_selenium(date_str)
            if len(df) > 0:
                fn_date = current_date.strftime("%m_%d_%Y")
                out_file = os.path.join(year_path, f'strikeout_odds_{fn_date}.parquet')
                df.to_parquet(out_file)
                print(f"Saved: {out_file}")

            current_date += datetime.timedelta(days=1)
