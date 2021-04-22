import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

for dirname, _, filenames in os.walk('F:/school/Papers/timeseriesNew/TS-Net/dataset/SP500/individual_stocks_5yr/individual_stocks_5yr/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


sns.set_style('darkgrid')

aapl = pd.read_csv('F:/school/Papers/timeseriesNew/TS-Net/dataset/SP500/individual_stocks_5yr/individual_stocks_5yr//AAPL_data.csv')
print(aapl.head())
print(aapl.shape)
print(aapl.info())
aapl['date'] = pd.to_datetime(aapl.date)