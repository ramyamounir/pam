import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd
from glob import glob
import textwrap

import sys;sys.path.append('./')
from src.visualizations.configs import method_names, method_ids, method_colors

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'


def get_dfs(files):
    dfs = []
    results = {}
    for file in sorted(files):
        method = file.split('/')[-3]
        if method not in results: results[method] = []
        corr = float(file.split('/')[-2])
        data = json.load(open(file, 'r'))
        results[method].append(data['mse_imgs'])

    df = pd.DataFrame(results)
    df = df.drop(columns=['HN-1-50'])
    df = df.rename(columns=method_names)
    df.index = [0.0, 0.2, 0.4]
    return df


dfs = [get_dfs(glob(f'results/videos_noisy/run_003/clevrer/*/*/{str(seed).zfill(3)}.json')) for seed in range(10)]

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

plt.xticks(mean_df.index)
plt.title('CLEVRER Image Reconstruction MSE')
plt.xlabel('% of active bits changed')
plt.ylabel('Mean Squared Error')
# plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/videos_noisy_mse.svg', format='svg', dpi=600, bbox_inches='tight')


