import matplotlib.pyplot as plt
import seaborn as sns

import sys;sys.path.append('./')
from src.visualizations.configs import method_names, method_ids, method_colors

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

def create_legend(methods, path, ncol=None):

    if ncol==None:
        ncol = len(methods)

    fig_legend = plt.figure(figsize=(20, 1))
    for i in methods:
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
    plt.legend(handles, labels, loc='center', ncol=ncol, bbox_to_anchor=(0.5, 2.0), borderaxespad=0, frameon=True)

    plt.savefig(path, format='svg', dpi=600, bbox_inches='tight')


create_legend(['PAM-4', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'PC-2', 'HN-1-5', 'HN-2-5', 'HN-1-50', 'HN-2-50'], './results/figures/results_1_legend.svg')
create_legend(['PAM-1', 'PAM-4', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'HN-1-50', 'HN-2-50'], './results/figures/results_2_legend.svg')
create_legend(['PAM-4', 'PC-1', 'PC-2', 'HN-2-50'], './results/figures/app_noise_legend.svg')
create_legend(['PAM-1', 'PAM-4', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'HN-1-50', 'HN-2-50'], './results/figures/app_online_sdr_forgetting_legend.svg', ncol=4)
create_legend(['PAM-1', 'PAM-4', 'PAM-8', 'PAM-16', 'PAM-24', 'PC-1', 'HN-1-5', 'HN-2-5', 'HN-1-50', 'HN-2-50'], './results/figures/app_capacity_scale.svg', ncol=5)
