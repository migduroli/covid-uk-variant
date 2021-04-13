import pandas as pd

from matplotlib import pyplot as plt
from pandas import DataFrame


def setup_matplotlib(
        font_family: str = 'serif',
        font_size: int = 8,
        fig_size: tuple = (5, 3),
        fig_dpi: int = 100
):
    plt.rc('font', family=font_family)
    plt.rc('font', size=font_size)
    plt.rcParams['figure.figsize'] = fig_size
    plt.rcParams['figure.dpi'] = fig_dpi
    plt.rc('text', usetex=True)


def find_percentile(data: DataFrame, col: str, n: int, eps: float=0.01):
    y = data[col]
    cum_sum = y.cumsum()
    z = y[(cum_sum >= (n*(1-eps)/100)*cum_sum.max()) & (cum_sum <= (n*(1+eps)/100)*cum_sum.max())]
    idx = z.index.tolist()
    return data.iloc[idx]


df = pd.read_csv('data/kidd-rtqpcr.csv', sep=';')

m_neg = find_percentile(df, 's_neg', 50)
m_pos = find_percentile(df, 's_pos', 50)
setup_matplotlib()

fig, ax = plt.subplots()
ax.bar(df.ct_value, df.s_neg, color='red', alpha=0.35, width=0.6)
ax.bar(df.ct_value, df.s_pos, color='blue', alpha=0.35, width=0.6)
ax.plot([m_neg.ct_value, m_neg.ct_value], [0, 20], '--', color='red')
ax.plot([m_pos.ct_value, m_pos.ct_value], [0, 20], '--', color='blue')
# ax.text(m_neg.ct_value-1, 12,
#         f'{m_neg.ct_value.to_list()[0]}',
#         color='red',
#         rotation=90)
# ax.text(m_pos.ct_value-1, 12,
#         f'{m_pos.ct_value.to_list()[0]}',
#         color='blue',
#         rotation=90)
ax.set_xlim([8, 40])
ax.set_ylim([0, 18])

ax.set_xlabel(r"$C_T$ value")
fig.tight_layout()
plt.savefig('figs/ct-value.pdf')