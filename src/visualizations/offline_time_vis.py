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
        df = df.drop(columns=['PAM-1', 'HN-2-50'])
        df = df.rename(columns=method_names)
        df.index = [10, 50, 100]
        dfs.append(df)
    return dfs


df = get_dfs(glob('results/offline_time/run_002/*.json'))[0]


# Plot the DataFrame
sns.set(style="darkgrid", context="talk")  # Set the style
plt.figure(figsize=(6, 6))  # Set the figure size

for column in df.columns:
    sns.lineplot(data=df[column], marker='o', markersize=10, dashes=False, color=method_colors[column], label=column)

plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.yscale('log')
plt.xticks(df.index)
plt.title('Time Vs Sequence Length')
plt.xlabel('Sequence Length')
plt.ylabel('Time (s)')
plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/offline_capacity_N.svg', format='svg', dpi=600, bbox_inches='tight')



quit()
# Get legend in a separate plot
sns.lineplot(x=[0], y=[0], marker='o', markersize=10, dashes=False, color=method_colors["tPC $L=2$"], label="tPC $L=2$") # adding extra label
handles, labels = plt.gca().get_legend_handles_labels()
sns.set(style="white", context="talk")
fig_legend = plt.figure(figsize=(20, 1))
fig_legend = plt.gca()
fig_legend.set_xticks([])
fig_legend.set_yticks([])
fig_legend.set_axis_off()
fig_legend.spines['top'].set_visible(False)
fig_legend.spines['right'].set_visible(False)
fig_legend.spines['bottom'].set_visible(False)
fig_legend.spines['left'].set_visible(False)
plt.legend(handles, labels, loc='center', ncol=len(handles), bbox_to_anchor=(0.5, 0.0), borderaxespad=0, frameon=True)

# plt.show()
plt.savefig('./results/figures/legend.svg', format='svg', dpi=600, bbox_inches='tight')
