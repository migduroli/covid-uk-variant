import pandas as pd

from matplotlib import pyplot as plt
from pandas import DataFrame


def find_percentile(data: DataFrame, col: str, n: int, eps: float=0.01):
    y = data[col]
    cum_sum = y.cumsum()
    z = y[(cum_sum >= (n*(1-eps)/100)*cum_sum.max()) & (cum_sum <= (n*(1+eps)/100)*cum_sum.max())]
    idx = z.index.tolist()
    return data.iloc[idx]


df = pd.read_csv('data/kidd-rtqpcr.csv', sep=';')

plt.bar(df.ct_value, df.s_pos, color='blue', alpha=0.35, width=0.6)
plt.bar(df.ct_value, df.s_neg, color='red', alpha=0.35, width=0.6)
