import pandas as pd
import pybaseball as pyb
import utils

a = pyb.rosters(2019)
utils.pdf(a.tail(10))