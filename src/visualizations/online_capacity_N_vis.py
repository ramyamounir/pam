import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd
from glob import glob

import sys;sys.path.append('./')
from src.visualizations.configs import method_names, method_ids, method_colors

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'


def get_dfs(files):
    dfs = []
    for file in files:
        data = json.load(open(file, 'r'))
        df = pd.DataFrame(data)
        df = df.rename(columns=method_names)
        df.index = range(10, 110, 10)
        dfs.append(df)
    return dfs


dfs = get_dfs(glob('results/offline_capacity_N/run_004/*.json'))

mean_df = pd.concat(dfs).groupby(level=0).mean()
std_df = pd.concat(dfs).groupby(level=0).std()


# Plot the DataFrame
sns.set(style="darkgrid", context="talk")  # Set the style
plt.figure(figsize=(6, 6))  # Set the figure size

for column in mean_df.columns:
    sns.lineplot(data=mean_df[column], marker='o', markersize=10, dashes=False, color=method_colors[column], label=column)
    plt.fill_between(mean_df.index, mean_df[column]-std_df[column], mean_df[column] + std_df[column], alpha=0.2, color=method_colors[column])

plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.yscale('log')
# plt.ylim([2,2000])
plt.xticks(mean_df.index)
plt.title('Online Sequence Capacity Vs. Input Size')
plt.xlabel(r'$N_c$')
plt.ylabel(r'$T_{max}$')
plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/online_capacity_N.svg', format='svg', dpi=600, bbox_inches='tight')


