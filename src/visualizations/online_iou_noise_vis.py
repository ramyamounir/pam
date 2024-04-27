import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob

import sys;sys.path.append('./')
from src.visualizations.configs import method_names, method_ids, method_colors


def get_dfs(files):
    dfs = []
    for file in files:
        data = json.load(open(file, 'r'))
        df = pd.DataFrame(data)
        df = df.drop(columns=['HN-1-50', 'PAM-8'])
        df = df.rename(columns=method_names)
        df.index = [0, 20, 40, 60, 80, 100]
        dfs.append(df)
    return dfs


dfs = get_dfs(glob('results/online_iou_noise/run_002/results_200_0.0_*.json'))

mean_df = pd.concat(dfs).groupby(level=0).mean()
std_df = pd.concat(dfs).groupby(level=0).std()


# Plot the DataFrame
sns.set(style="darkgrid", context="talk")  # Set the style
plt.figure(figsize=(6, 6))  # Set the figure size

for column in mean_df.columns:
    sns.lineplot(data=mean_df[column], marker='o', markersize=10, dashes=False, color=method_colors[column], label=column)
    plt.fill_between(mean_df.index, mean_df[column]-std_df[column], mean_df[column]+std_df[column], alpha=0.2, color=method_colors[column])


plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.xticks(mean_df.index)
plt.ylim([-0.5, 1.1])
plt.title('IoU Vs. Noise\nT=10\ncorrelation=0.0')
plt.xlabel('% active bits changed')
plt.ylabel('Normalized IoU')
plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/online_noise.svg', format='svg', dpi=600, bbox_inches='tight')





# for p in [10, 100]:
#     for b in [0.0, 0.3, 0.5]:

#         dfs = get_dfs(glob(f'results/online_iou_noise/run_002/results_{p}_{b}_*.json'))

#         mean_df = pd.concat(dfs).groupby(level=0).mean()
#         std_df = pd.concat(dfs).groupby(level=0).std()


#         # Plot the DataFrame
#         sns.set(style="darkgrid", context="talk")  # Set the style
#         plt.figure(figsize=(6, 6))  # Set the figure size

#         for column in mean_df.columns:
#             sns.lineplot(data=mean_df[column], marker='o', markersize=10, dashes=False, color=method_colors[column], label=column)
#             plt.fill_between(mean_df.index, mean_df[column]-std_df[column], mean_df[column]+std_df[column], alpha=0.2, color=method_colors[column])


#         plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
#         plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
#         plt.grid(which='both', linestyle='--')

#         plt.xticks(mean_df.index)
#         plt.ylim([-0.5, 1.1])
#         plt.title(f'Sequence Length={p}\nCorrelation={str(b)}')
#         plt.xlabel('% active bits changed')
#         plt.ylabel('Normalized IoU')
#         plt.legend().set_visible(False)

#         plt.savefig(f'./results/figures/online_noise_{p}_{str(b)}.svg', format='svg', dpi=600, bbox_inches='tight')


