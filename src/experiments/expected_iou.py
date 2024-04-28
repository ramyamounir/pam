import torch
import os, json
import numpy as np
from tqdm import tqdm

import sys; sys.path.append('./')
from src.utils.sdr import SDR

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


results = {
        "Empirical_0.5":[],
        "Analytical_0.5":[],
        "Empirical_0.1":[],
        "Analytical_0.1":[]}

p_range = [0.1, 0.5]
q_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for p in p_range:
    for q in q_range:

        ious = 0.0
        for _ in range(1000):

            a = SDR(N=100, S=int(p*100))
            b = SDR(N=100, S=int(q*100))
            ious += a.iou(b)

        results[f"Empirical_{p}"].append(ious/1000)
        results[f"Analytical_{p}"].append((p*q)/(p+q-(p*q)))

df = pd.DataFrame(results)


# Plot the DataFrame
sns.set(style="darkgrid", context="talk")  # Set the style
plt.figure(figsize=(8, 6))  # Set the figure size

sns.lineplot(x=q_range, y=results['Analytical_0.1'],label='Analytical p=0.1')
sns.lineplot(x=q_range, y=results['Empirical_0.1'], label='Empirical p=0.1', linestyle='--')

sns.lineplot(x=q_range, y=results['Analytical_0.5'],label='Analytical p=0.5')
sns.lineplot(x=q_range, y=results['Empirical_0.5'], label='Empirical p=0.5', linestyle='--')


plt.tick_params(axis='x', length=10, width=2, direction='inout', which='both')
plt.tick_params(axis='y', length=10, width=2, direction='inout', which='both')
plt.grid(which='both', linestyle='--')

plt.title('Analytical Vs. Empirical Expected IoU')
plt.xlabel('q')
plt.ylabel('Expected IoU')
# plt.show()  # Show the plot
plt.savefig('./results/figures/expected_iou.svg', format='svg', dpi=600, bbox_inches='tight')



