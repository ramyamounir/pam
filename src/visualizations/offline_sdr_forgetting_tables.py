import torch
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator
import seaborn as sns
import pandas as pd
from glob import glob
import numpy as np

import sys;sys.path.append('./')
from src.visualizations.configs import method_names, method_ids, method_colors

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'



def create_dfs(data):
    results = {}
    for k, v in data.items():
        v = v[2]
        v_len = len(v)
        new_v = []
        for vi in v: 
            v_copy = []
            for vi2 in vi[:-1]: v_copy.append(vi2)
            while len(v_copy) < v_len: v_copy.append(-1)
            new_v.append(v_copy)

        results[k] = new_v

    for k, v in results.items():
        results[k] = pd.DataFrame(v)

    return results


def average_dfs(dfs):
    names = list(dfs[0].keys())
    results = {name:{'mean': {}, 'std':{}} for name in names}

    for k in results.keys():
        vals = [dfs[i][k] for i in range(len(dfs))]
        results[k]['mean'] = pd.concat(vals).groupby(level=0).mean()
        results[k]['std'] = pd.concat(vals).groupby(level=0).std()

    return results

def get_dfs(files):
    dfs_full = []
    for file in files:
        data = json.load(open(file, 'r'))
        dfs = create_dfs(data)
        dfs_full.append(dfs)

    return dfs_full


dfs = get_dfs(glob('results/offline_sdr_forgetting/run_002/results_10_*.json'))
avg_dfs = average_dfs(dfs)

for k, v in avg_dfs.items():
    mean = v['mean'].copy()
    mean = mean.applymap(lambda x: "{:.3f}".format(float(x)) if x != "-" else "-")

    std = v['std'].copy()
    std = std.applymap(lambda x: "{:.3f}".format(float(x)) if x != "-" else "-")

    # create a new DataFrame with the combined data
    df_combined = mean.copy()
    df_combined = df_combined.apply(lambda x: x.astype(str) + ' ± ' + std[x.name].astype(str))

    # create a mask for the upper triangle and diagonal
    mask = np.logical_not(np.triu(np.ones(df_combined.shape), k=0))

    # replace the values in the DataFrame with "-" where the mask is True
    df_combined = df_combined.where(mask, "-")

    # reset index and columns
    df_combined.index = range(1, 11)
    df_combined.columns = range(1, 11)

    # convert the combined DataFrame to a LaTeX table
    latex_table = df_combined.to_latex(index=False)
    print(k)
    print(latex_table)
    print('\n'*5)


