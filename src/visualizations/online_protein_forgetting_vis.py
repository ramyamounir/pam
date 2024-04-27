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

def create_bwt_df(data, ix):

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
        results[k] = calc_bwt(v)

    return pd.DataFrame(results, index=[ix])


def get_dfs(files):
    dfs = []
    for file, ix in zip(files, [5, 10, 15, 20]):
        data = json.load(open(file, 'r'))
        dfs.append(create_bwt_df(data, ix))

    dfs = pd.concat(dfs)
    dfs = dfs.drop(columns=['PAM-4', 'PAM-8'])
    dfs = dfs.rename(columns=method_names)

    return dfs

dfs = []
for num_seq in ['05', '10', '15', '20']:
    df = get_dfs(glob(f'results/offline_protein_forgetting/run_003/results_{num_seq}_*.json'))
    dfs.append(df)

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
plt.title('Backward Transfer Vs. Protein Sequences')
plt.xlabel('Number of Protein Sequences')
plt.ylabel('Backward Transfer (BWT)')
plt.legend().set_visible(False)

# plt.show()  # Show the plot
plt.savefig('./results/figures/online_protein_forgetting.svg', format='svg', dpi=600, bbox_inches='tight')

