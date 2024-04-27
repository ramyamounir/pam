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
    for file in files:
        data = json.load(open(file, 'r'))
        results[file.split('/')[-2]] = data['mse_imgs']

    df = pd.DataFrame(results, index=[0])
    df = df.rename(columns=method_names)
    return df


df = pd.concat([get_dfs(glob(f'results/videos_cl/run_001/moving_mnist/*/{str(seed).zfill(3)}.json')) for seed in range(10)])

df_melt = pd.melt(df, var_name='method', value_name='value')
# method_order = ['PAM $N_k=8$', 'tPC',  'tPC $L=2$',  'AHN d=1, W=$0.5N_c$',  'AHN d=2, W=$0.5N_c$' ]
method_order = ['PAM $N_k=8$', 'tPC',   'AHN d=1, W=$0.5N_c$',  'tPC $L=2$',  'AHN d=2, W=$0.5N_c$' ]
df_melt['method'] = pd.Categorical(df_melt['method'], categories=method_order, ordered=True)


# Plot the DataFrame
sns.set(style="darkgrid", context="talk")  # Set the style
plt.figure(figsize=(10, 6))  # Set the figure size

sns.boxplot(data=df_melt, x='method', y='value', palette=method_colors, showfliers=False)

# Wrap the method names to appear on two lines
plt.xticks([i for i in range(len(method_order))], 
           [textwrap.fill(method, width=wl) for method, wl in zip(method_order, [10, 5, 10, 5, 10])])


plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.title('Moving MNIST Image Reconstruction Error')
plt.xlabel('')
plt.ylabel('Mean Squared Error')

# plt.show()  # Show the plot
plt.savefig('./results/figures/videos_cl_mse.svg', format='svg', dpi=600, bbox_inches='tight')


