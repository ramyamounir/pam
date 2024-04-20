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

def create_bwt_df(data):

    def calc_bwt(results):
        counter = 0
        s = 0.0
        for i in results:
            if len(i) == 1: continue
            s+= sum(i[:-1])
            counter += len(i[:-1])

        return s/counter

    results = {}
    for k, v in data.items():
        results[k] = []
        for b in range(len(v)):
            results[k].append(calc_bwt(v[b]))

    return pd.DataFrame(results)

def get_dfs(files):
    dfs = []
    for file in files:
        data = json.load(open(file, 'r'))
        df = create_bwt_df(data)
        df = df.rename(columns=method_names)
        df.index = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        dfs.append(df)
    return dfs


dfs = get_dfs(glob('results/offline_sdr_forgetting/run_002/results_50_*.json'))

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
plt.title('Backward Transfer Vs. Correlation')
plt.xlabel('Correlation')
plt.ylabel('Backward Transfer (BWT)')
plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/sdr_forgetting.svg', format='svg', dpi=600, bbox_inches='tight')



# quit()
# Get legend in a separate plot

fig_legend = plt.figure(figsize=(20, 1))
include = ['PAM-4', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'PC-2', 'HN-1-5', 'HN-2-5', 'HN-1-50', 'HN-2-50']
for i in include:
    name = method_names[i]
    sns.lineplot(x=[0], y=[0], marker='o', markersize=10, dashes=False, color=method_colors[name], label=name) # adding extra label
handles, labels = plt.gca().get_legend_handles_labels()
sns.set(style="white", context="talk")
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
plt.savefig('./results/figures/legend2.svg', format='svg', dpi=600, bbox_inches='tight')
