from tabulate import tabulate_formats, tabulate
import os
import pandas as pd
import sys
import time


def pdf(df):
    print(tabulate(df, headers='keys', tablefmt=tabulate_formats[2]))


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} created.')
    # else:
    #     print(f'Directory {dir} already exists.')


def grab_data(path, start_year, end_year):
    # print(f'Pulling data from path: "{path}", start_year: {start_year}, end_year: {end_year}.')
    all_data = []
    yrs = list(range(start_year, end_year+1))

    for filename in os.listdir(path):
        if int(filename.split('_')[1].split('.')[0]) in yrs:
            # print(filename)
            if filename.endswith(".parquet"):
                file_path = os.path.join(path, filename)
                try:
                    df = pd.read_parquet(file_path)
                    all_data.append(df)
                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        print("No data found in the directory.")
        return None


def oh_waiter(secs,desc=""):
    for i in range(secs, 0, -1):
        sys.stdout.write(f"\rWaiting {i} seconds...")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write(f"\rWaiting 0 seconds... Done {desc}!\n")


def pinch_cols(df):
    df.columns = [
        "_".join([str(level) for level in col_tuple if level])
        for col_tuple in df.columns.values
    ]
