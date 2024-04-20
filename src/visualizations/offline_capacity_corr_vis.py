import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd
from glob import glob

import sys;sys.path.append('./')
from src.visualizations.configs import method_names, method_ids, method_colors


def get_dfs(files):
    dfs = []
    for file in files:
        data = json.load(open(file, 'r'))
        df = pd.DataFrame(data)
        df = df.drop(columns=['HN-1-5', 'HN-2-5'])
        df = df.rename(columns=method_names)
        df.index = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        dfs.append(df)
    return dfs


dfs = get_dfs(glob('results/offline_capacity_corr/run_002/*.json'))

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
plt.ylim([2,2000])
plt.xticks(mean_df.index)
plt.title('Sequence Capacity Vs. Correlation')
plt.xlabel('Correlation')
plt.ylabel(r'$T_{max}$')
plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/offline_capacity_corr.svg', format='svg', dpi=600, bbox_inches='tight')



