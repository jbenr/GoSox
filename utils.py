from tabulate import tabulate_formats, tabulate

def pdf(df):
    print(tabulate(df, headers='keys', tablefmt=tabulate_formats[2]))